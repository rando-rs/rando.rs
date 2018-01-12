use super::*;
use rand;
use rand::Rng;

quickcheck! {
    fn const_1() -> () {
        let sl = &[1];
        let mut it = sl.rand_iter();

        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), None);
    }

    fn const_2() -> () {
        let sl = &[1, 1];
        let mut it = sl.rand_iter();

        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), None);
    }

    fn vec_1(v: Vec<u32>) -> () {
        assert_eq_up_to_order(v.rand_iter(), v.iter());
    }

    fn std_rng_1(v: Vec<u32>) -> () {
        assert_eq_up_to_order(v.rand_iter().with_rng(rand::StdRng::new().unwrap()), v.iter());
    }

    fn assert_eq_up_to_order_1(vec_1: Vec<u32>) -> () {
        let mut vec_2 = vec_1.clone();

        rand::thread_rng().shuffle(&mut vec_2);

        assert_eq_up_to_order(vec_1, vec_2);
    }
}
