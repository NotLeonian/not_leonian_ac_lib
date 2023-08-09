//! <https://github.com/NotLeonian/not_leonian_ac_lib>

/// 1つ以上の変数を1行で出力するマクロ
#[allow(unused_macros)]
macro_rules! outputln {
    ($var:expr) => {
        println!("{}",$var);
    };
    ($var:expr,$($vars:expr),+) => {
        print!("{} ",$var);
        outputln!($($vars),+);
    };
}

/// 1つの配列またはベクターの中身を1行で出力するマクロ
#[allow(unused_macros)]
macro_rules! outputvec {
    ($var:expr) => {
        for i in 0..($var.len()-1) {
            print!("{} ",$var[i]);
        }
        println!("{}",$var[$var.len()-1]);
    };
}

/// 条件によって変わる1行を出力するマクロ（条件、真の場合、偽の場合の順）
#[allow(unused_macros)]
macro_rules! outputif {
    ($cond:expr,$true:expr,$false:expr) => {
        if $cond {
            println!("{}",$true);
        } else {
            println!("{}",$false);
        }
    };
}

/// 条件によって"Yes"または"No"の1行を出力するマクロ
#[allow(unused_macros)]
macro_rules! outputyn {
    ($cond:expr) => {
        outputif!($cond,"Yes","No");
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
