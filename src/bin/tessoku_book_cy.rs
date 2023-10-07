#[allow(unused_attributes)] #[macro_use] #[allow(unused_imports)] use not_leonian_ac_lib::*;

fn main() {
    /// 標準入力のマクロ（インタラクティブ問題ならば中のマクロを変更）
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
    }

    input!(n:usize);
    let sieve=sieve_of_eratosthenes(n);
    for i in 2..=n {
        if sieve[i] {
            outputln!(i);
        }
    }
}
