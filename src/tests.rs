use super::*;
use rand;
use rand::Rng;
use rand::SeedableRng;

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

    fn with_rng_1(v: Vec<u32>) -> () {
        assert_eq_up_to_order(
            v.rand_iter().with_rng(rand::StdRng::from_rng(rand::EntropyRng::new()).unwrap()),
            v.iter(),
        );
    }

    fn with_rng_2(v: Vec<u32>) -> () {
        let mut rng = rand::StdRng::from_rng(rand::EntropyRng::new()).unwrap();

        assert_eq_up_to_order(
            v.rand_iter().with_rng(&mut rng),
            v.iter(),
        );
    }

    fn assert_eq_up_to_order_1(vec_1: Vec<u32>) -> () {
        let mut vec_2 = vec_1.clone();

        rand::thread_rng().shuffle(&mut vec_2);

        assert_eq_up_to_order(vec_1, vec_2);
    }
}

#[test]
fn check_version_numbers_in_readme_deps() {
    assert_markdown_deps_updated!("README.mkd");
}

#[test]
fn check_docs_html_root_url() {
    assert_html_root_url_updated!("src/lib.rs");
}
