#[allow(unused_attributes)] #[macro_use] #[allow(unused_imports)] use not_leonian_ac_lib::*;
use proconio::marker::Usize1;

fn main() {
    /// 標準入力のマクロ（インタラクティブ問題ならば中のマクロを変更）
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
    }

    input!(n:usize,q:usize,a:[usize;n],lr:[(Usize1,usize);q]);
    let s=a.construct_prefix_sum();
    for (l,r) in lr {
        outputln!(s.calculate_partial_sum(l, r));
    }
}
