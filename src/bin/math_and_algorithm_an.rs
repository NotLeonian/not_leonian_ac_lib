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

    input!(n:usize,m:usize,ab:[(Usize1,Usize1);m]);
    let g=construct_graph::<VecGraph>(n, m, &ab);
    let dist=g.dist_of_shortest_paths(0, false);
    for v in 0..n {
        dist[v].output_val_or(usize::MAX);
    }
}
