#[allow(unused_attributes)] #[macro_use] #[allow(unused_imports)] use not_leonian_ac_lib::*;

fn main() {
    /// 標準入力のマクロ（インタラクティブ問題ならば中のマクロを変更）
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
    }

    input!(q:usize,x:[usize;q]);
    for x in x {
        output_yes_or_no(x.is_prime());
    }
}
