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

    input!(h:usize,w:usize,x:[[usize;w];h],q:usize,abcd:[(Usize1,Usize1,usize,usize);q]);
    let s=x.construct_2d_prefix_sum();
    for (a,b,c,d) in abcd {
        outputln!(s.calculate_2d_partial_sum(a, b, c, d));
    }
}
