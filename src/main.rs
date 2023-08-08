//! <https://github.com/NotLeonian/not_leonian_ac_lib>

fn main() {
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
    }
    input!();
}
