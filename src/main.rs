//! <https://github.com/NotLeonian/not_leonian_ac_lib>

/// 1つ以上の変数を1行で出力するマクロ
#[allow(unused_macros)]
macro_rules! outputln {
    ($var:expr) => {
        {
            println!("{}",$var);
        }
    };
    ($var:expr,$($vars:expr),+) => {
        {
            print!("{} ",$var);
            outputln!($($vars),+);
        }
    };
}

/// 1つの配列またはベクターの中身を1行で出力するマクロ
#[allow(unused_macros)]
macro_rules! outputvec {
    ($var:expr) => {
        {
            let vec=$var;
            let len=vec.len();
            for (i,&v) in vec.iter().enumerate() {
                if i<len-1 {
                    print!("{} ",v);
                } else {
                    println!("{}",v);
                }
            }
        }
    };
}

/// 1つの2次元配列または2次元ベクターの中身を複数行で出力するマクロ
#[allow(unused_macros)]
macro_rules! output2dvec {
    ($var:expr) => {
        {
            for vec in &$var {
                outputvec!(vec);
            }
        }
    };
}

/// 条件によって変わる1行を出力するマクロ（条件、真の場合、偽の場合の順）
#[allow(unused_macros)]
macro_rules! outputif {
    ($cond:expr,$true:expr,$false:expr) => {
        {
            if $cond {
                println!("{}",$true);
            } else {
                println!("{}",$false);
            }
        }
    };
}

/// 条件によって"Yes"または"No"の1行を出力するマクロ
#[allow(unused_macros)]
macro_rules! outputyn {
    ($cond:expr) => {
        {
            outputif!($cond,"Yes","No");
        }
    };
}

/// 「n重のmoveつきクロージャ (;大域変数;) n組の範囲」でベクターを生成するマクロ（トリッキーなマクロなので使用には注意）
#[allow(unused_macros)]
macro_rules! vecrange {
    ($func:expr,$begin:expr,$end:expr) => {
        {
            ($begin..$end).map(|i| $func(i)).collect::<Vec<_>>()
        }
    };
    ($func:expr;$($vals:expr),+;$begin:expr,$end:expr) => {
        {
            ($begin..$end).map(|i| $func($($vals),+,i)).collect::<Vec<_>>()
        }
    };
    ($func:expr,$begin:expr,$end:expr,$($ranges:expr),+) => {
        {
            ($begin..$end).map(|i| vecrange!($func(i),$($ranges),+)).collect::<Vec<_>>()
        }
    };
    ($func:expr;$($vals:expr),+;$begin:expr,$end:expr,$($ranges:expr),+) => {
        {
            ($begin..$end).map(|i| vecrange!($func($($vals),+,i),$($ranges),+)).collect::<Vec<_>>()
        }
    };
}

fn main() {
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
        // インタラクティブ問題の場合、proconio::input!マクロを削除→
        // proconio::input_interactive!マクロのコメントアウトを解除
    }
    input!();
}

/// 累積和のベクターを構築する関数（ModIntなどに使う）
#[allow(dead_code)]
fn construct_algebraic_prefix_sum<T>(array: &Vec::<T>, prefix_sum: &mut Vec::<T>, zero: T) where T: Copy, T: std::ops::Add<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(array.len()+1==prefix_sum.len());
    prefix_sum[0]=zero;
    for i in 0..array.len() {
        prefix_sum[i+1]=prefix_sum[i]+array[i];
    }
}

/// 累積和のベクターを構築する関数（整数型などに使う）
#[allow(dead_code)]
fn construct_prefix_sum<T>(array: &Vec::<T>, prefix_sum: &mut Vec::<T>) where T: Copy, T: std::ops::Add<Output=T>, T: num_traits::Zero<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    construct_algebraic_prefix_sum(array, prefix_sum, num_traits::zero());
}

/// 構築した累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
#[allow(dead_code)]
fn calculate_partial_sum<T>(prefix_sum: &Vec::<T>, l: usize, r: usize) -> T where T: Copy, T: std::ops::Sub<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(l < prefix_sum.len() && r < prefix_sum.len());
    return prefix_sum[r+1]-prefix_sum[l];
}

/// 2次元累積和のベクターを構築する関数（ModIntなどに使う）
#[allow(dead_code)]
fn construct_algebraic_2d_prefix_sum<T>(array: &Vec::<Vec::<T>>, prefix_sum: &mut Vec::<Vec::<T>>, zero: T) where T: Copy, T: std::ops::Add<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(array.len()+1==prefix_sum.len());
    prefix_sum[0][0]=zero;
    for i in 0..array.len() {
        assert!(array[i].len()+1==prefix_sum[i].len() && array[i].len()==array[0].len());
        for j in 0..array[i].len() {
            prefix_sum[i+1][j+1]=prefix_sum[i+1][j]+array[i][j];
        }
    }
    for j in 0..array[0].len() {
        for i in 0..array.len() {
            prefix_sum[i+1][j+1]=prefix_sum[i][j+1]+prefix_sum[i+1][j+1];
        }
    }
}

/// 2次元累積和のベクターを構築する関数（整数型などに使う）
#[allow(dead_code)]
fn construct_2d_prefix_sum<T>(array: &Vec::<Vec::<T>>, prefix_sum: &mut Vec::<Vec::<T>>) where T: Copy, T: std::ops::Add<Output=T>, T: num_traits::Zero<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    construct_algebraic_2d_prefix_sum(array, prefix_sum, num_traits::zero());
}

/// 構築した2次元累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
#[allow(dead_code)]
fn calculate_2d_partial_sum<T>(prefix_sum: &Vec::<Vec::<T>>, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> T where T: Copy, T: std::ops::Add<Output=T>, T: std::ops::Sub<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(l_i < prefix_sum.len() && l_j < prefix_sum[0].len() && r_i < prefix_sum.len() && r_j < prefix_sum[0].len());
    return prefix_sum[r_i+1][r_j+1]-prefix_sum[r_i+1][l_j]-prefix_sum[l_i][r_j+1]+prefix_sum[l_i][l_j];
}

/// 数字の列strをusize型のbase進数の数値に変換する関数
#[allow(dead_code)]
fn radix_converse(str: &String, base: usize) -> usize {
    usize::from_str_radix(&str, base as u32).unwrap()
}
