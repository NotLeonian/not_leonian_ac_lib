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

/// 配列やベクターの中身を1行で出力するトレイト
trait Outputln {
    /// 配列やベクターの中身を1行で出力する関数
    fn outputln(&self);
}

impl<T> Outputln for Vec::<T> where T: std::fmt::Display {
    fn outputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<self.len()-1 {
                print!("{} ",&var);
            } else {
                println!("{}",&var);
            }
        }
    }
}

impl<T, const N: usize> Outputln for [T;N] where T: Sized, T: std::fmt::Display {
    fn outputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<self.len()-1 {
                print!("{} ",&var);
            } else {
                println!("{}",&var);
            }
        }
    }
}

/// 配列やベクターの中身を複数行で出力するトレイト
trait Outputlines {
    /// 配列やベクターの中身を複数行で出力する関数
    fn outputlines(&self);
}

impl<T> Outputlines for Vec::<T> where T: Outputln {
    fn outputlines(&self) {
        for v in self {
            v.outputln();
        }
    }
}

impl<T, const N: usize> Outputlines for [T;N] where T: Sized, T: Outputln {
    fn outputlines(&self) {
        for v in self {
            v.outputln();
        }
    }
}

/// 条件によって変わる1行を出力するトレイト
trait Outputif {
    /// 条件によって変わる1行を出力する関数（引数は順に真の場合、偽の場合の出力）
    fn outputif<T1,T2>(self, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display;
    /// 条件によって"Yes"または"No"の1行を出力する関数
    fn outputln(self);
}

impl Outputif for bool {
    fn outputif<T1,T2>(self, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display {
        if self {
            println!("{}",ok);
        } else {
            println!("{}",bad);
        }
    }
    fn outputln(self) {
        self.outputif("Yes", "No");
    }
}

/// ベクターの先頭にfilledを追加してmだけ右にずらす関数のトレイト
trait MoveRight where Self: std::ops::Index<usize> {
    /// ベクターの先頭にfilledを追加してmだけ右にずらす関数
    fn move_right(&mut self, m: usize, filled: Self::Output);
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

/// chminとchmaxのトレイト
trait Chminmax {
    /// challengerのほうが大きければchallengerで上書きする関数
    fn chmax(&mut self, challenger: Self);
    /// challengerのほうが小さければchallengerで上書きする関数
    fn chmin(&mut self, challenger: Self);
}

impl<T> Chminmax for T where T: Copy, T: std::cmp::PartialOrd {
    fn chmax(&mut self, challenger: Self) {
        if challenger>*self {
            *self=challenger;
        }
    }
    fn chmin(&mut self, challenger: Self) {
        if challenger<*self {
            *self=challenger;
        }
    }
}

/// chminとchmaxのトレイト
trait ChminmaxVec where Self: std::ops::Index<usize> {
    /// ベクターの中身について、challengerのほうが大きければchallengerで上書きする関数
    fn chmax_vec(&mut self, index: usize, challenger: Self::Output);
    /// ベクターの中身について、challengerのほうが小さければchallengerで上書きする関数
    fn chmin_vec(&mut self, index: usize, challenger: Self::Output);
}

impl<T> ChminmaxVec for Vec::<T> where T: Copy, T: std::cmp::PartialOrd {
    fn chmax_vec(&mut self, index: usize, challenger: T) {
        if challenger>self[index] {
            self[index]=challenger;
        }
    }
    fn chmin_vec(&mut self, index: usize, challenger: T) {
        if challenger<self[index] {
            self[index]=challenger;
        }
    }
}

impl<T, const N: usize> ChminmaxVec for [T;N] where T: Copy, T: std::cmp::PartialOrd {
    fn chmax_vec(&mut self, index: usize, challenger: T) {
        if challenger>self[index] {
            self[index]=challenger;
        }
    }
    fn chmin_vec(&mut self, index: usize, challenger: T) {
        if challenger<self[index] {
            self[index]=challenger;
        }
    }
}

/// 添字についてchminとchmaxを行うトレイト
trait ChminmaxIndex<T> where T: std::ops::Index<Self>, T::Output: std::cmp::PartialOrd {
    /// vecの現在の添字の値よりchallengerの値のほうが大きければchallengerで上書きする関数
    fn chmax_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが小さければchallengerで上書きする関数
    fn chmin_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが小さくなければchallengerで上書きする関数
    fn chmaxeq_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが大きくなければchallengerで上書きする関数
    fn chmineq_index(&mut self, vec: &T, challenger: Self);
}

impl<T> ChminmaxIndex<T> for usize where T: std::ops::Index<Self>, T::Output: std::cmp::PartialOrd {
    fn chmax_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]>vec[*self] {
            *self=challenger;
        }
    }
    fn chmin_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]<vec[*self] {
            *self=challenger;
        }
    }
    fn chmaxeq_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]>=vec[*self] {
            *self=challenger;
        }
    }
    fn chmineq_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]<=vec[*self] {
            *self=challenger;
        }
    }
}

struct VecGraph {
    adj_list: Vec<Vec<(usize,usize)>>,
    vertex_info: Vec::<usize>
}

struct SetGraph {
    adj_list: Vec::<std::collections::BTreeSet::<(usize,usize)>>,
    vertex_info: Vec::<usize>
}

/// グラフについてのトレイト ((usize,usize)の2次元ベクターと(usize,usize)のBTreeSetのベクターについて実装)
trait Graph where Self: Sized {
    /// グラフを初期化する関数
    fn new(n: usize) -> Self;
    /// 頂点数を返す関数
    fn size(&self) -> usize;
    /// 辺を追加する関数
    fn push(&mut self, a: usize, b: usize, w: usize);
    /// 点に数の情報を追加する関数
    fn inform(&mut self, c: &Vec::<usize>);
    /// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_graph(n: usize, m: usize, ab: &Vec::<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
            g.push(b, a, 1);
        }
        return g;
    }
    /// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_directed_graph(n: usize, m: usize, ab: &Vec::<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
        }
        return g;
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_graph(n: usize, m: usize, abw: &Vec::<(usize,usize,usize)>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
            g.push(b, a, w);
        }
        return g;
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec::<(usize,usize,usize)>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
        }
        return g;
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧、点の数の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_informed_graph(n: usize, m: usize, abw: &Vec::<(usize,usize,usize)>, c: &Vec::<usize>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
            g.push(b, a, w);
        }
        g.inform(c);
        return g;
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧、点の数の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_informed_directed_graph(n: usize, m: usize, abw: &Vec::<(usize,usize,usize)>, c: &Vec::<usize>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
        }
        g.inform(c);
        return g;
    }
    /// DFSの関数
    fn dfs<F1,Fd,F2,F3,F4>(&mut self, start: usize, preorder: F1, adj_determine: Fd, inorder: F2, already: F3, postorder: F4) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F4: FnMut(usize,&mut Vec::<usize>) -> bool;
    /// BFSの関数
    fn bfs<F1,Fd,F2,F3>(&mut self, start: usize, preorder: F1, adj_determine: Fd, inorder: F2, already: F3) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool;
}

impl Graph for VecGraph {
    fn new(n: usize) -> Self {
        return VecGraph { adj_list: vec![Vec::<(usize,usize)>::new();n], vertex_info: vec![0usize;n] };
    }
    fn size(&self) -> usize {
        return self.adj_list.len();
    }
    fn push(&mut self, a: usize, b: usize, w: usize) {
        self.adj_list[a].push((b,w));
    }
    fn inform(&mut self, c: &Vec::<usize>) {
        self.vertex_info=c.clone();
    }
    fn dfs<F1,Fd,F2,F3,F4>(&mut self, start: usize, mut preorder: F1, mut adj_determine: Fd, mut inorder: F2, mut already: F3, mut postorder: F4) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F4: FnMut(usize,&mut Vec::<usize>) -> bool {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut stack=vec![start+n,start];
        while let Some(v)=stack.pop() {
            if v<n {
                if !preorder(v,&mut self.vertex_info) {
                    return false;
                }
                for &(u,w) in &self.adj_list[v] {
                    if !adj_determine(v,u,w,&mut self.vertex_info) {
                        continue;
                    }
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u+n);
                        stack.push(u);
                        if !inorder(v,u,w,&mut self.vertex_info) {
                            return false;
                        }
                    } else {
                        if !already(v,u,w,&mut self.vertex_info) {
                            return false;
                        }
                    }
                }
            } else {
                let v=v-n;
                if !postorder(v,&mut self.vertex_info) {
                    return false;
                }
            }
        }
        return true;
    }
    fn bfs<F1,Fd,F2,F3>(&mut self, start: usize, mut preorder: F1, mut adj_determine: Fd, mut inorder: F2, mut already: F3) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut queue=std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(v)=queue.pop_front() {
            if !preorder(v,&mut self.vertex_info) {
                return false;
            }
            for &(u,w) in &self.adj_list[v] {
                if !adj_determine(v,u,w,&mut self.vertex_info) {
                    continue;
                }
                if !seen[u] {
                    seen[u]=true;
                    queue.push_back(u);
                    if !inorder(v,u,w,&mut self.vertex_info) {
                        return false;
                    }
                } else {
                    if !already(v,u,w,&mut self.vertex_info) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}

impl Graph for SetGraph {
    fn new(n: usize) -> Self {
        return SetGraph{ adj_list: vec![std::collections::BTreeSet::<(usize,usize)>::new();n], vertex_info: vec![0usize;n] };
    }
    fn size(&self) -> usize {
        return self.adj_list.len();
    }
    fn push(&mut self, a: usize, b: usize, w: usize) {
        self.adj_list[a].insert((b,w));
    }
    fn inform(&mut self, c: &Vec::<usize>) {
        self.vertex_info=c.clone();
    }
    fn dfs<F1,Fd,F2,F3,F4>(&mut self, start: usize, mut preorder: F1, mut adj_determine: Fd, mut inorder: F2, mut already: F3, mut postorder: F4) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F4: FnMut(usize,&mut Vec::<usize>) -> bool {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut stack=vec![start+n,start];
        while let Some(v)=stack.pop() {
            if v<n {
                if !preorder(v,&mut self.vertex_info) {
                    return false;
                }
                for &(u,w) in &self.adj_list[v] {
                    if !adj_determine(v,u,w,&mut self.vertex_info) {
                        continue;
                    }
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u+n);
                        stack.push(u);
                        if !inorder(v,u,w,&mut self.vertex_info) {
                            return false;
                        }
                    } else {
                        if !already(v,u,w,&mut self.vertex_info) {
                            return false;
                        }
                    }
                }
            } else {
                let v=v-n;
                if !postorder(v,&mut self.vertex_info) {
                    return false;
                }
            }
        }
        return true;
    }
    fn bfs<F1,Fd,F2,F3>(&mut self, start: usize, mut preorder: F1, mut adj_determine: Fd, mut inorder: F2, mut already: F3) -> bool where F1: FnMut(usize,&mut Vec::<usize>) -> bool, Fd: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F2: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool, F3: FnMut(usize,usize,usize,&mut Vec::<usize>) -> bool {
        let n=self.size();
        assert!(start<n);
        let mut seen=vec![false;n];
        seen[start]=true;
        let mut queue=std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(v)=queue.pop_front() {
            if !preorder(v,&mut self.vertex_info) {
                return false;
            }
            for &(u,w) in &self.adj_list[v] {
                if !adj_determine(v,u,w,&mut self.vertex_info) {
                    continue;
                }
                if !seen[u] {
                    seen[u]=true;
                    queue.push_back(u);
                    if !inorder(v,u,w,&mut self.vertex_info) {
                        return false;
                    }
                } else {
                    if !already(v,u,w,&mut self.vertex_info) {
                        return false;
                    }
                }
            }
        }
        return true;
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

/// 累積和についてのトレイト
trait PrefixSum {
    /// 累積和のベクターを構築する関数
    fn construct_prefix_sum(array: &Self) -> Self;
    /// 構築した累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
    fn calculate_partial_sum(&self, l: usize, r: usize) -> Self::Output where Self: std::ops::Index<usize>;
}

impl<T> PrefixSum for Vec::<T> where T: Copy, T: std::ops::Add<Output=T>, T: std::ops::Sub<Output=T> {
    fn construct_prefix_sum(array: &Self) -> Self {
        let mut prefix_sum=vec![array[0]-array[0];array.len()+1];
        for i in 0..array.len() {
            prefix_sum[i+1]=prefix_sum[i]+array[i];
        }
        return prefix_sum;
    }
    fn calculate_partial_sum(&self, l: usize, r: usize) -> <Self as std::ops::Index::<usize>>::Output {
        assert!(l < self.len() && r < self.len());
        return self[r+1]-self[l];
    }
}

/// 2次元累積和についてのトレイト
trait TwoDPrefixSum {
    /// 2次元累積和のベクターを構築する関数
    fn construct_2d_prefix_sum(array: &Self) -> Self;
    /// 構築した2次元累積和のベクターから部分和を計算する関数（0-indexedの閉区間）
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <Self::Output as std::ops::Index::<usize>>::Output where Self: std::ops::Index::<usize>, Self::Output: std::ops::Index::<usize>;
}

impl<T> TwoDPrefixSum for Vec::<Vec::<T>> where T: Copy, T: std::ops::Add<Output=T>, T: std::ops::Sub<Output=T> {
    fn construct_2d_prefix_sum(array: &Self) -> Self {
        let mut prefix_sum=vec![vec![array[0][0]-array[0][0];array[0].len()+1];array.len()+1];
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
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <<Self as std::ops::Index::<usize>>::Output as std::ops::Index::<usize>>::Output {
        assert!(l_i < self.len() && l_j < self[0].len() && r_i < self.len() && r_j < self[0].len());
        return self[r_i+1][r_j+1]-self[r_i+1][l_j]-self[l_i][r_j+1]+self[l_i][l_j];
    }
}

/// 単一の文字を数値に変換する関数のトレイト
trait FromChar {
    /// 文字を数値に変換する関数（0-indexedの閉区間）
    fn from_foo(self, begin: char, end: char) -> usize;
    /// 数字（文字）を1桁の数値に変換する関数
    fn from_fig(self) -> usize;
    /// 大文字のアルファベットを1桁の数値に変換する関数（0-indexed）
    fn from_uppercase(self) -> usize;
    /// 小文字のアルファベットを1桁の数値に変換する関数（0-indexed）
    fn from_lowercase(self) -> usize;
}

impl FromChar for char {
    fn from_foo(self, begin: char, end: char) -> usize {
        if self>=begin && self<=end {
            return self as usize-begin as usize;
        } else {
            panic!();
        }
    }
    fn from_fig(self) -> usize {
        return self.from_foo('0', '9');
    }
    fn from_uppercase(self) -> usize {
        return self.from_foo('A', 'Z');
    }
    fn from_lowercase(self) -> usize {
        return self.from_foo('a', 'z');
    }
}

/// 数値を単一の文字に変換する関数のトレイト
trait ToChar {
    /// 数値を文字に変換する関数（0-indexed）
    fn to_foo(self, begin: char, width: usize) -> char;
    /// 1桁の数値を数字（文字）に変換する関数
    fn to_fig(self) -> char;
    /// 1桁の数値を大文字のアルファベットに変換する関数（0-indexed）
    fn to_uppercase(self) -> char;
    /// 1桁の数値を小文字のアルファベットに変換する関数（0-indexed）
    fn to_lowercase(self) -> char;
}

impl ToChar for usize {
    fn to_foo(self, begin: char, width: usize) -> char {
        if self<width {
            return char::from_u32(self as u32+begin as u32).unwrap();
        } else {
            panic!();
        }
    }
    fn to_fig(self) -> char {
        return self.to_foo('0', 10);
    }
    fn to_uppercase(self) -> char {
        return self.to_foo('A', 26);
    }
    fn to_lowercase(self) -> char {
        return self.to_foo('a', 26);
    }
}

/// 最小値を取り出すことのできる優先度つきキューの構造体
#[allow(dead_code)]
struct RevBinaryHeap<T> where T: Ord {
    binary_heap: std::collections::BinaryHeap<std::cmp::Reverse<T>>
}

impl<T> RevBinaryHeap<T> where T: Ord {
    #[allow(dead_code)]
    fn new() -> RevBinaryHeap<T> {
        RevBinaryHeap { binary_heap: std::collections::BinaryHeap::<std::cmp::Reverse<T>>::new() }
    }
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        return self.binary_heap.is_empty();
    }
    #[allow(dead_code)]
    fn len(&self) -> usize {
        return self.binary_heap.len();
    }
    #[allow(dead_code)]
    fn push(&mut self, item: T) {
        self.binary_heap.push(std::cmp::Reverse(item));
    }
    #[allow(dead_code)]
    fn pop(&mut self) -> Option<T> {
        if !self.is_empty() {
            let std::cmp::Reverse(ret)=self.binary_heap.pop().unwrap();
            return Some(ret);
        } else {
            return None;
        }
    }
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.binary_heap.clear();
    }
}

/// 数の文字列の10進法への変換についてのトレイト
trait ToDecimal {
    /// radix進法の数の文字列を10進数の数値へ変換する関数
    fn to_decimal(&self, radix: usize) -> usize;
}

impl ToDecimal for String {
    fn to_decimal(&self, radix: usize) -> usize {
        return usize::from_str_radix(&self, radix as u32).unwrap();
    }
}
