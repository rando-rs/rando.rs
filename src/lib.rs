#![deny(missing_docs)]
#![deny(warnings)]
#![doc(html_root_url = "https://docs.rs/rando/0.1.0")]
#![doc(test(attr(deny(warnings))))]
//! A library for iteration in random order.
//!
//! # Examples
//!
//! ```rust
//! # extern crate rand;
//! # extern crate rando;
//! use rand::StdRng;
//! use rando::Rando;
//! use rando::assert_eq_up_to_order;
//! # fn main() {
//! # (|| -> ::std::io::Result<()> {
//!
//! assert_eq_up_to_order(&[1, 2, 3], [1, 2, 3].rand_iter());
//!
//! assert_eq_up_to_order(&['a', 'b', 'c'], ['c', 'a', 'b'].rand_iter());
//!
//! let primes = [2, 3, 5, 7, 11];
//! let mut p2 = Vec::new();
//!
//! primes.rand_iter().for_each(|n| p2.push(n));
//!
//! assert_eq_up_to_order(&primes, p2);
//!
//! // These random number generators have the same seeds...
//! let rng_1 = StdRng::new()?;
//! let rng_2 = rng_1.clone();
//!
//! // ...so `RandIter`s using them should iterate in the same order.
//! assert_eq!(
//!     primes.rand_iter().with_rng(rng_1).collect::<Vec<_>>(),
//!     primes.rand_iter().with_rng(rng_2).collect::<Vec<_>>()
//! );
//! # Ok(())
//! # })().unwrap()
//! # }
//! ```

extern crate get_trait;
extern crate iter_trait;
extern crate len_trait;
extern crate rand;
extern crate smallvec;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
#[macro_use]
extern crate version_sync;

use get_trait::Get;
use iter_trait::HasMapData;
use len_trait::Len;
use rand::distributions::IndependentSample;
use rand::distributions::Range;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::ops::Deref;

#[cfg(test)]
mod tests;

/// By default, a [`RandIter`] will be able to iterate over this number of items without allocating
/// memory on the heap.
///
/// [`RandIter`]: <struct.RandIter.html>
pub const DEFAULT_MEM_LEN: usize = 32;

type DefaultMemory<T> = SmallVec<[T; DEFAULT_MEM_LEN]>;

/// A trait for collections over which this library allows one to iterate in **rand**om **o**rder.
pub trait Rando: Get + Len + HasMapData
where
    Self::Key: PartialEq,
{
    /// Returns an iterator that iterates over this collection in random order.
    #[inline]
    fn rand_iter(&self) -> RandIter<Self>;
}

impl<T: ?Sized> Rando for T
where
    T: Get + Len + HasMapData,
    T::Key: PartialEq,
{
    #[inline]
    fn rand_iter(&self) -> RandIter<Self> {
        RandIter {
            collection: self,
            rng: rand::thread_rng(),
            range: Default::default(),
            memory: Default::default(),
        }
    }
}

/// An iterator over the items in a collection, in random order.
///
/// This `struct` is created by the [`rand_iter`] method of the [`Rando`] trait, which this library
/// implements for some common collection types.
///
/// [`Rando`]: <trait.Rando.html>
/// [`rand_iter`]: <trait.Rando.html#method.rand_iter>
#[derive(Debug)]
pub struct RandIter<
    'coll,
    Collection: ?Sized,
    Mem = DefaultMemory<<Collection as HasMapData>::Key>,
    Rng = rand::ThreadRng,
> where
    Collection: 'coll + Get + Len + HasMapData,
    Mem: Memory<Collection::Key>,
    Rng: rand::Rng,
{
    collection: &'coll Collection,
    // TODO: Once specialization is stable, use () as the default `Rng` type, and retrieve the
    // `thread_rng` only as needed.
    rng: Rng,
    range: Option<Range<usize>>,
    memory: Mem,
}

impl<'coll, Collection: ?Sized, Mem, Rng> RandIter<'coll, Collection, Mem, Rng>
where
    Collection: 'coll
        + Get
        + Len
        + HasMapData,
    Mem: Memory<Collection::Key>,
    Rng: rand::Rng,
{
    /// Sets the random number generator for the [`RandIter`].
    ///
    /// The random number generator must implement the [`Rng`] trait from the [`rand`] crate.
    ///
    /// The random number generator defaults to that returned by [`rand::thread_rng`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate rando;
    /// # extern crate rand;
    /// # use rando::Rando;
    /// # use rando::assert_eq_up_to_order;
    /// # fn main() {
    /// # (|| -> ::std::io::Result<()> {
    /// use rand::StdRng;
    ///
    /// let primes = [2, 3, 5, 7, 11];
    ///
    /// assert_eq_up_to_order(
    ///     primes.rand_iter(),
    ///     primes.rand_iter().with_rng(StdRng::new()?),
    /// );
    /// # Ok(())
    /// # })().unwrap()
    /// # }
    /// ```
    ///
    /// [`RandIter`]: <struct.RandIter.html>
    /// [`rand`]: <https://docs.rs/rand/*/rand/>
    /// [`rand::thread_rng`]: <https://docs.rs/rand/*/rand/fn.thread_rng.html>
    /// [`Rng`]: <https://docs.rs/rand/*/rand/trait.Rng.html>
    #[inline]
    pub fn with_rng<NewRng>(self, rng: NewRng) -> RandIter<'coll, Collection, Mem, NewRng>
    where
        NewRng: rand::Rng,
    {
        let RandIter {
            collection,
            rng: _,
            range,
            memory,
        } = self;

        RandIter {
            collection,
            rng,
            range,
            memory,
        }
    }

/// Sets the type of memory buffer for the [`RandIter`].
///
/// The memory buffer is a data-structure in which this [`RandIter`] will store values
/// representing the indices or other keys of the items that it has already yielded. To be such
/// a memory buffer, a type must implement the [`Memory`] trait. This library implements this
/// trait for a few types:
///
/// - Servo's [`SmallVec`] — The default type of memory buffer is a [`SmallVec`] capable of
/// holding [`DEFAULT_MEM_LEN`] entries without allocation, and unbounded entries with
/// allocation.
///
/// - The standard [`Vec`] — This kind of memory buffer may be more appropriate than the
/// default for iterating over a collection strongly expected to contain more than
/// [`DEFAULT_MEM_LEN`] entries.
///
/// - The standard [`BTreeSet`] — This kind of memory buffer may be more appropriate than the
/// default for iterating over a large collection; specifically, one large enough that
/// performing linear search on an unsorted list of its keys becomes more expensive than
/// keeping those keys in a sorted [`BTreeSet`] and performing binary search on this sorted
/// data-structure.
///
/// # Examples
///
/// ```rust
/// # extern crate rando;
/// # extern crate smallvec;
/// # use rando::DEFAULT_MEM_LEN;
/// # use rando::Rando;
/// # use rando::assert_eq_up_to_order;
/// # use smallvec::SmallVec;
/// # use std::collections::BTreeSet;
/// # fn main() {
/// let primes = [2, 3, 5, 7, 11];
///
/// assert_eq_up_to_order(
///     primes.rand_iter(),
///     primes.rand_iter().with_memory::<SmallVec<[_; DEFAULT_MEM_LEN * 2]>>(),
/// );
///
/// assert_eq_up_to_order(
///     primes.rand_iter(),
///     primes.rand_iter().with_memory::<Vec<_>>(),
/// );
///
/// assert_eq_up_to_order(
///     primes.rand_iter(),
///     primes.rand_iter().with_memory::<BTreeSet<_>>(),
/// );
/// # }
/// ```
///
/// [`RandIter`]: <struct.RandIter.html>
/// [`Memory`]: <trait.Memory.html>
/// [`Vec`]: <https://doc.rust-lang.org/std/vec/struct.Vec.html>
/// [`BTreeSet`]: <https://doc.rust-lang.org/std/collections/struct.BTreeSet.html>
/// [`SmallVec`]: <https://docs.rs/smallvec/*/smallvec/struct.SmallVec.html>
/// [`DEFAULT_MEM_LEN`]: <constant.DEFAULT_MEM_LEN.html>
    #[inline]
    pub fn with_memory<NewMem>(self) -> RandIter<'coll, Collection, NewMem, Rng>
    where
        NewMem: Memory<Collection::Key>,
    {
        let RandIter {
            collection,
            rng,
            range,
            memory: _,
        } = self;

        RandIter {
            collection,
            rng,
            range,
            memory: Default::default(),
        }
    }
}

impl<'coll, Collection: ?Sized, Mem, Rng> Iterator
    for RandIter<'coll, Collection, Mem, Rng>
where
    Collection: Get + Len + HasMapData<Key = usize>,
    Mem: Memory<Collection::Key>,
    Rng: rand::Rng,
{
    type Item = &'coll Collection::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let k = choose_key(
            self.collection,
            &mut self.rng,
            &mut self.range,
            &mut self.memory,
        )?;

        let v = self.collection.get(&k).expect(
            "`rando`: Internal error: \
             `choose_key` chose an invalid key",
        );

        Some(v)
    }
}

#[inline]
fn choose_key<'coll, Collection: ?Sized, Mem, Rng>(
    collection: &'coll Collection,
    rng: &mut Rng,
    range: &mut Option<Range<usize>>,
    keys_already_chosen: &mut Mem,
) -> Option<usize>
where
    Collection: Get + Len + HasMapData<Key = usize>,
    Mem: Memory<Collection::Key>,
    Rng: rand::Rng,
{
    if keys_already_chosen.len() == collection.len() {
        return None;
    }

    if range.is_none() && collection.len() > 0 {
        *range = Some(Range::new(0, collection.len()));
    }

    let range = match range {
        &mut Some(ref r) => r,
        &mut None => return None,
    };

    loop {
        let k = range.ind_sample(rng);

        if !keys_already_chosen.contains(&k) {
            keys_already_chosen.push(k);
            return Some(k);
        }
    }
}

/// A trait for data-structures in which a [`RandIter`] can store values of type `K` representing
/// the indices or other keys of the items that it has already yielded.
///
/// [`RandIter`]: <struct.RandIter.html>
pub trait Memory<K>: Default {
    /// Returns the number of `K` values stored in `self`.
    #[inline]
    fn len(&self) -> usize;

    /// Returns `true` if `key` has been stored in `self`, and `false` otherwise.
    #[inline]
    fn contains(&self, key: &K) -> bool;

    /// Stores `key` in `self`.
    #[inline]
    fn push(&mut self, key: K);
}

impl<K, A> Memory<K> for SmallVec<A>
where
    K: PartialEq,
    A: smallvec::Array<Item = K>,
{
    #[inline]
    fn len(&self) -> usize {
        self.deref().len()
    }

    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.deref().contains(key)
    }

    #[inline]
    fn push(&mut self, key: K) {
        SmallVec::push(self, key)
    }
}

impl<K> Memory<K> for Vec<K>
where
    K: PartialEq,
{
    #[inline]
    fn len(&self) -> usize {
        self.deref().len()
    }

    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.deref().contains(key)
    }

    #[inline]
    fn push(&mut self, key: K) {
        Vec::push(self, key)
    }
}

impl<K> Memory<K> for BTreeSet<K>
where
    K: Ord,
{
    #[inline]
    fn len(&self) -> usize {
        BTreeSet::len(self)
    }

    #[inline]
    fn contains(&self, key: &K) -> bool {
        BTreeSet::contains(self, key)
    }

    #[inline]
    fn push(&mut self, key: K) {
        BTreeSet::insert(self, key);
    }
}


/// Asserts that the two given iterators (or values that can be converted into iterators) yield the
/// same items, without regard to the order in which those items are yielded.
///
/// To try to be clearer, duplicate items are not counted as being "the same".
///
/// # Panics
///
/// This function panics if the two iterators do not yield the same items, without regard to order.
///
/// # Examples
///
/// ```rust
/// # use rando::assert_eq_up_to_order;
/// assert_eq_up_to_order(&[1, 2, 3], &[3, 2, 1]);
///
/// assert_eq_up_to_order(&[4, 5, 6], &[4, 6, 5]);
///
/// assert_eq_up_to_order(&['a', 'b', 'c'], &['c', 'a', 'b']);
/// ```
#[inline]
pub fn assert_eq_up_to_order<I1, I2, Item>(i1: I1, i2: I2)
where
    I1: IntoIterator<Item = Item>,
    I2: IntoIterator<Item = Item>,
    Item: Ord + Debug,
{
    let mut counts_1 = BTreeMap::<Item, usize>::new();
    let mut counts_2 = BTreeMap::<Item, usize>::new();

    #[inline]
    fn count<I, Item>(it: I, counts: &mut BTreeMap<Item, usize>)
    where
        I: IntoIterator<Item = Item>,
        Item: Ord,
    {
        for item in it {
            *counts.entry(item).or_insert(0) += 1;
        }
    }

    count(i1, &mut counts_1);
    count(i2, &mut counts_2);

    assert_eq!(counts_1, counts_2);
}
