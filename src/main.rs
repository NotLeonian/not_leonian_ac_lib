//! <https://github.com/NotLeonian/not_leonian_ac_lib>

fn main() {
    /// インタラクティブ問題の場合、proconio::input!マクロからproconio::input_interactive!マクロへ変更
    macro_rules! input {
        ($($tt:tt)*) => {
            proconio::input!($($tt)*);
            // proconio::input_interactive!($($tt)*);
        };
    }

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

    input!();
}

/// 1つのベクターの中身を順に1行で出力する関数
#[allow(dead_code)]
fn output_vec<T>(vec: &Vec::<T>) where T: std::fmt::Display {
    for (i,var) in vec.iter().enumerate() {
        if i<vec.len()-1 {
            print!("{} ",&var);
        } else {
            println!("{}",&var);
        }
    }
}

/// 1つの2次元ベクターの中身を順に複数行で出力する関数
#[allow(dead_code)]
fn output_2d_vec<T>(vecs: &Vec::<Vec::<T>>) where T: std::fmt::Display {
    for v in vecs {
        for (i,var) in v.iter().enumerate() {
            if i<v.len()-1 {
                print!("{} ",&var);
            } else {
                println!("{}",&var);
            }
        }
    }
}

/// 条件によって変わる1行を出力する関数（引数は条件の関数、真の場合の文字列、偽の場合の文字列の順）
#[allow(dead_code)]
fn output_if<F,T>(determine: F, ok: T, bad: T) where F: Fn()->bool, T: std::fmt::Display {
    if determine() {
        println!("{}",ok);
    } else {
        println!("{}",bad);
    }
}

/// 条件によって"Yes"または"No"の1行を出力する関数（引数は条件の関数）
#[allow(dead_code)]
fn output_yes_or_no<F>(determine: F) where F: Fn() -> bool {
    output_if(determine, "Yes", "No");
}

/// for文風にbeginからendまでの結果を格納したベクターを生成する関数（0-indexedの左閉右開区間）
#[allow(dead_code)]
fn vec_range<N,F,T>(begin: N, end: N, func: F) -> Vec::<T> where std::ops::Range<N>: Iterator, F: Fn(<std::ops::Range<N> as Iterator>::Item) -> T {
    return (begin..end).map(|i| func(i)).collect::<Vec::<T>>();
}

/// max関数
#[allow(dead_code)]
fn max<T>(left: T, right: T) -> T where T: std::cmp::PartialOrd {
    return if left>right {
        left
    } else {
        right
    };
}

/// min関数
#[allow(dead_code)]
fn min<T>(left: T, right: T) -> T where T: std::cmp::PartialOrd {
    return if left<right {
        left
    } else {
        right
    };
}

/// currentよりchallengerのほうが大きければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmax<T>(challenger: T, current: &mut T) where T: std::cmp::PartialOrd {
    if challenger>*current {
        *current=challenger;
    }
}

/// currentよりchallengerのほうが小さければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmin<T>(challenger: T, current: &mut T) where T: std::cmp::PartialOrd {
    if challenger<*current {
        *current=challenger;
    }
}

/// a[current]よりa[challenger]のほうが大きければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmax_index<T>(a: &Vec::<T>, challenger: usize, current: &mut usize) where T: std::cmp::PartialOrd {
    if a[challenger]>a[*current] {
        *current=challenger;
    }
}

/// a[current]よりa[challenger]のほうが小さければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmin_index<T>(a: &Vec::<T>, challenger: usize, current: &mut usize) where T: std::cmp::PartialOrd {
    if a[challenger]<a[*current] {
        *current=challenger;
    }
}

/// a[current]よりa[challenger]が小さくなければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmax_eq_index<T>(a: &Vec::<T>, challenger: usize, current: &mut usize) where T: std::cmp::PartialOrd {
    if a[challenger]>=a[*current] {
        *current=challenger;
    }
}

/// a[current]よりa[challenger]が大きくなければcurrentをchallengerで上書きする関数
#[allow(dead_code)]
fn chmin_eq_index<T>(a: &Vec::<T>, challenger: usize, current: &mut usize) where T: std::cmp::PartialOrd {
    if a[challenger]<=a[*current] {
        *current=challenger;
    }
}

/// グラフに関するtrait (Vec::<Vec::<usize>>とVec::<BTreeSet::<usize>>について実装)
trait Graph where Self: Sized {
    /// グラフを初期化する関数
    fn new(n: usize) -> Self;
    /// 頂点数を返す関数
    fn size(&self) -> usize;
    /// 辺を追加する関数
    fn push(&mut self, a: usize, b: usize);
    /// 子を見る関数
    fn see_child<F>(&self, v: usize, seen: &mut Vec::<bool>, func: F) where F: FnMut(usize);
    /// 無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_undirected_graph(n: usize, m: usize, ab: &Vec::<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b);
            g.push(b, a);
        }
        return g;
    }
    /// 有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_directed_graph(n: usize, m: usize, ab: &Vec::<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b);
        }
        return g;
    }
    /// DFSの関数
    fn dfs<F1,F2,F3>(&self, start: usize, mut preorder: F1, mut inorder: F2, mut postorder: F3) where F1: FnMut(usize), F2: FnMut(usize,usize), F3: FnMut(usize) {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut stack=vec![n+start,start];
        while !stack.is_empty() {
            let v=stack.pop().unwrap();
            if v<n {
                preorder(v);
                self.see_child(v, &mut seen, |u| {
                    stack.push(u);
                    inorder(v,u);
                });
            } else {
                let v=v-n;
                postorder(v);
            }
        }
    }
    /// 帰りがけ順を省略するDFSの関数
    fn dfs_pre_and_in<F1,F2>(&self, start: usize, mut preorder: F1, mut inorder: F2) where F1: FnMut(usize), F2: FnMut(usize,usize) {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut stack=vec![start];
        while !stack.is_empty() {
            let v=stack.pop().unwrap();
            preorder(v);
            self.see_child(v, &mut seen, |u| {
                stack.push(u);
                inorder(v,u);
            });
        }
    }
    /// BFSの関数
    fn bfs<F1,F2>(&self, start: usize, mut preorder: F1, mut inorder: F2) where F1: FnMut(usize), F2: FnMut(usize,usize) {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut queue=std::collections::VecDeque::new();
        queue.push_back(start);
        while !queue.is_empty() {
            let v=queue.pop_front().unwrap();
            preorder(v);
            self.see_child(v, &mut seen, |u| {
                queue.push_back(u);
                inorder(v,u);
            });
        }
    }
}

impl Graph for Vec::<Vec::<usize>> {
    fn new(n: usize) -> Self {
        return vec![Vec::<usize>::new();n];
    }
    fn size(&self) -> usize {
        return self.len();
    }
    fn push(&mut self, a: usize, b: usize) {
        self[a].push(b);
    }
    fn see_child<F>(&self, v: usize, seen: &mut Vec::<bool>, mut func: F) where F: FnMut(usize) {
        for &u in &self[v] {
            if !seen[u] {
                seen[u]=true;
                func(u);
            }
        }
    }
}

impl Graph for Vec::<std::collections::BTreeSet::<usize>> {
    fn new(n: usize) -> Self {
        return vec![std::collections::BTreeSet::<usize>::new();n];
    }
    fn size(&self) -> usize {
        return self.len();
    }
    fn push(&mut self, a: usize, b: usize) {
        self[a].insert(b);
    }
    fn see_child<F>(&self, v: usize, seen: &mut Vec::<bool>, mut func: F) where F: FnMut(usize) {
        for &u in &self[v] {
            if !seen[u] {
                seen[u]=true;
                func(u);
            }
        }
    }
}

/// 二分探索の関数（整数）
#[allow(dead_code)]
fn binary_search<F>(ok: isize, bad: isize, determine: F) -> isize where F: Fn(isize) -> bool {
    let right=ok>bad;
    let mut ok=ok;
    let mut bad=bad;
    while if right {
        ok-bad>1
    } else {
        bad-ok>1
    } {
        let mid=(ok+bad)/2;
        if determine(mid) {
            ok=mid;
        } else {
            bad=mid;
        }
    }
    return ok;
}

/// 二分探索の関数（浮動小数点数）
#[allow(dead_code)]
fn float_binary_search<F>(ok: f64, bad: f64, determine: F, rerror: f64) -> f64 where F: Fn(f64) -> bool {
    let right=ok>bad;
    let mut ok=ok;
    let mut bad=bad;
    while if right {
        ok-bad>ok*rerror
    } else {
        bad-ok>ok*rerror
    } {
        let mid=(ok+bad)/2.;
        if determine(mid) {
            ok=mid;
        } else {
            bad=mid;
        }
    }
    return ok;
}

/// 累積和のベクターを構築する関数（ModIntなどに使う）
#[allow(dead_code)]
fn construct_algebraic_prefix_sum<T>(array: &Vec::<T>, zero: T) -> Vec::<T> where T: Copy, T: std::ops::Add<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    let mut prefix_sum=vec![zero;array.len()+1];
    for i in 0..array.len() {
        prefix_sum[i+1]=prefix_sum[i]+array[i];
    }
    return prefix_sum;
}

/// 累積和のベクターを構築する関数（整数型などに使う）
#[allow(dead_code)]
fn construct_prefix_sum<T>(array: &Vec::<T>) -> Vec::<T> where T: Copy, T: std::ops::Add<Output=T>, T: num_traits::Zero<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    return construct_algebraic_prefix_sum(array, num_traits::zero());
}

/// 構築した累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
#[allow(dead_code)]
fn calculate_partial_sum<T>(prefix_sum: &Vec::<T>, l: usize, r: usize) -> T where T: Copy, T: std::ops::Sub<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(l < prefix_sum.len() && r < prefix_sum.len());
    return prefix_sum[r+1]-prefix_sum[l];
}

/// 2次元累積和のベクターを構築する関数（ModIntなどに使う）
#[allow(dead_code)]
fn construct_algebraic_2d_prefix_sum<T>(array: &Vec::<Vec::<T>>, zero: T) -> Vec::<Vec::<T>> where T: Copy, T: std::ops::Add<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    let mut prefix_sum=vec![vec![zero;array[0].len()+1];array.len()+1];
    prefix_sum[0][0]=zero;
    for i in 0..array.len() {
        assert!(array[i].len()==array[0].len());
        for j in 0..array[0].len() {
            prefix_sum[i+1][j+1]=prefix_sum[i+1][j]+array[i][j];
        }
    }
    for j in 0..array[0].len() {
        for i in 0..array.len() {
            prefix_sum[i+1][j+1]=prefix_sum[i][j+1]+prefix_sum[i+1][j+1];
        }
    }
    return prefix_sum;
}

/// 2次元累積和のベクターを構築する関数（整数型などに使う）
#[allow(dead_code)]
fn construct_2d_prefix_sum<T>(array: &Vec::<Vec::<T>>) -> Vec::<Vec::<T>> where T: Copy, T: std::ops::Add<Output=T>, T: num_traits::Zero<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    return construct_algebraic_2d_prefix_sum(array, num_traits::zero());
}

/// 構築した2次元累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
#[allow(dead_code)]
fn calculate_2d_partial_sum<T>(prefix_sum: &Vec::<Vec::<T>>, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> T where T: Copy, T: std::ops::Add<Output=T>, T: std::ops::Sub<Output=T>, Vec<T>: std::ops::Index<usize, Output=T> {
    assert!(l_i < prefix_sum.len() && l_j < prefix_sum[0].len() && r_i < prefix_sum.len() && r_j < prefix_sum[0].len());
    return prefix_sum[r_i+1][r_j+1]-prefix_sum[r_i+1][l_j]-prefix_sum[l_i][r_j+1]+prefix_sum[l_i][l_j];
}

/// 文字を数値に変換する関数（0-indexedの閉区間）
#[allow(dead_code)]
fn something_to_num(c: char, begin: char, end: char) -> usize {
    if c>=begin && c<=end {
        return c as usize-begin as usize;
    } else {
        panic!();
    }
}

/// 数値を文字に変換する関数（0-indexed）
#[allow(dead_code)]
fn something_from_num(n: usize, begin: char, width: usize) -> char {
    if n<width {
        return char::from_u32(n as u32+begin as u32).unwrap();
    } else {
        panic!();
    }
}

/// 数字（文字）を1桁の数値に変換する関数（0-indexed）
#[allow(dead_code)]
fn fig_to_num(c: char) -> usize {
    return something_to_num(c, '0', '9');
}

/// 1桁の数値を数字（文字）に変換する関数（0-indexed）
#[allow(dead_code)]
fn num_to_fig(n: usize) -> char {
    return something_from_num(n, '0', 10);
}

/// 大文字のアルファベットを1桁の数値に変換する関数（0-indexed）
#[allow(dead_code)]
fn uppercase_to_num(c: char) -> usize {
    return something_to_num(c, 'A', 'Z');
}

/// 1桁の数値を大文字のアルファベットに変換する関数（0-indexed）
#[allow(dead_code)]
fn num_to_uppercase(n: usize) -> char {
    return something_from_num(n, 'A', 26);
}

/// 小文字のアルファベットを1桁の数値に変換する関数（0-indexed）
#[allow(dead_code)]
fn lowercase_to_num(c: char) -> usize {
    return something_to_num(c, 'a', 'z');
}

/// 1桁の数値を小文字のアルファベットに変換する関数（0-indexed）
#[allow(dead_code)]
fn num_to_lowercase(n: usize) -> char {
    return something_from_num(n, 'a', 26);
}

/// 数字の列strをusize型のbase進数の数値に変換する関数
#[allow(dead_code)]
fn radix_converse(str: &String, base: usize) -> usize {
    usize::from_str_radix(&str, base as u32).unwrap()
}
