//! <https://github.com/NotLeonian/not_leonian_ac_lib>  
//!   
//! Copyright (c) 2023 Not_Leonian  
//! Released under the MIT license  
//! <https://opensource.org/licenses/mit-license.php>  

/// 1つ以上の値を1行で出力するマクロ
#[macro_export]
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

/// 配列やベクターの中身を1行で出力するトレイト
pub trait Outputln {
    /// 配列やベクターの中身を1行で出力する関数
    fn outputln(&self);
}

impl<T> Outputln for Vec<T> where T: std::fmt::Display {
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

impl<T> Outputln for [T] where T: std::fmt::Display {
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

impl<T, const N: usize> Outputln for [T;N] where T: Sized + std::fmt::Display {
    fn outputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<N-1 {
                print!("{} ",&var);
            } else {
                println!("{}",&var);
            }
        }
    }
}

/// 配列やベクターの中身を複数行で出力するトレイト
pub trait Outputlns {
    /// 配列やベクターの中身を複数行で出力する関数
    fn outputlns(&self);
}

impl<T> Outputlns for Vec<T> where T: Outputln {
    fn outputlns(&self) {
        for v in self {
            v.outputln();
        }
    }
}

impl<T> Outputlns for [T] where T: Outputln {
    fn outputlns(&self) {
        for v in self {
            v.outputln();
        }
    }
}

impl<T, const N: usize> Outputlns for [T;N] where T: Sized + Outputln {
    fn outputlns(&self) {
        for v in self {
            v.outputln();
        }
    }
}

/// 条件によって変わる1行を出力する関数（引数は順に条件と真の場合、偽の場合の出力）
#[allow(dead_code)]
pub fn outputif<T1,T2>(cond: bool, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display {
    if cond {
        println!("{}",ok);
    } else {
        println!("{}",bad);
    }
}

/// 条件によって"Yes"または"No"の1行を出力する関数
#[allow(dead_code)]
pub fn output_yes_or_no(cond: bool) {
    outputif(cond, "Yes", "No");
}

/// 存在すれば値を、存在しなければ-1を出力するトレイト
pub trait OutputValOr {
    /// 値がmaxより小さければ自身を出力し、maxであれば-1を出力する関数
    fn output_val_or(self, max: Self);
}

impl OutputValOr for usize {
    fn output_val_or(self, max: Self) {
        if self<max {
            println!("{}",self);
        } else {
            println!("-1");
        }
    }
}

/// 1つ以上の値を1行でstderrに出力するマクロ
#[macro_export]
#[allow(unused_macros)]
macro_rules! eoutputln {
    ($var:expr) => {
        eprintln!("{}",$var);
    };
    ($var:expr,$($vars:expr),+) => {
        eprint!("{} ",$var);
        eoutputln!($($vars),+);
    };
}

/// 配列やベクターの中身を1行でstderrに出力するトレイト
pub trait Eoutputln {
    /// 配列やベクターの中身を1行でstderrに出力する関数
    fn eoutputln(&self);
}

impl<T> Eoutputln for Vec<T> where T: std::fmt::Display {
    fn eoutputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<self.len()-1 {
                eprint!("{} ",&var);
            } else {
                eprintln!("{}",&var);
            }
        }
    }
}

impl<T> Eoutputln for [T] where T: std::fmt::Display {
    fn eoutputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<self.len()-1 {
                eprint!("{} ",&var);
            } else {
                eprintln!("{}",&var);
            }
        }
    }
}

impl<T, const N: usize> Eoutputln for [T;N] where T: Sized + std::fmt::Display {
    fn eoutputln(&self) {
        for (i,var) in self.iter().enumerate() {
            if i<N-1 {
                eprint!("{} ",&var);
            } else {
                eprintln!("{}",&var);
            }
        }
    }
}

/// 配列やベクターの中身を複数行でstderrに出力するトレイト
pub trait Eoutputlns {
    /// 配列やベクターの中身を複数行でstderrに出力する関数
    fn eoutputlns(&self);
}

impl<T> Eoutputlns for Vec<T> where T: Eoutputln {
    fn eoutputlns(&self) {
        for v in self {
            v.eoutputln();
        }
    }
}

impl<T> Eoutputlns for [T] where T: Eoutputln {
    fn eoutputlns(&self) {
        for v in self {
            v.eoutputln();
        }
    }
}

impl<T, const N: usize> Eoutputlns for [T;N] where T: Sized + Eoutputln {
    fn eoutputlns(&self) {
        for v in self {
            v.eoutputln();
        }
    }
}

/// 条件によって変わる1行をstderrに出力する関数（引数は順に条件と真の場合、偽の場合の出力）
#[allow(dead_code)]
pub fn eoutputif<T1,T2>(cond: bool, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display {
    if cond {
        eprintln!("{}",ok);
    } else {
        eprintln!("{}",bad);
    }
}

/// 条件によって"Yes"または"No"の1行をstderrに出力する関数
#[allow(dead_code)]
pub fn eoutput_yes_or_no(cond: bool) {
    eoutputif(cond, "Yes", "No");
}

/// 存在すれば値を、存在しなければ-1をstderrに出力するトレイト
pub trait EoutputValOr {
    /// 値がmaxより小さければ自身を出力し、maxであれば-1をstderrに出力する関数
    fn eoutput_val_or(self, max: Self);
}

impl EoutputValOr for usize {
    fn eoutput_val_or(self, max: Self) {
        if self<max {
            eprintln!("{}",self);
        } else {
            eprintln!("-1");
        }
    }
}

/// ベクターの先頭にfilledを追加してmだけ右にずらす関数のトレイト
pub trait MoveRight where Self: std::ops::Index<usize> {
    /// ベクターの先頭にfilledを追加してmだけ右にずらす関数
    fn move_right(&mut self, m: usize, filled: Self::Output);
}

impl<T> MoveRight for Vec<T> where T: Clone {
    fn move_right(&mut self, m: usize, filled: T) {
        let n=self.len();
        self.resize(n+m,filled.clone());
        for i in (0..n).rev() {
            self[i+m]=self[i].clone();
        }
        for i in 0..m {
            self[i]=filled.clone();
        }
    }
}

/// for文風にbeginからendまでの結果を格納したベクターを生成する関数（0-indexedの左閉右開区間）
#[allow(dead_code)]
pub fn vec_range<N,F,T>(begin: N, end: N, func: F) -> Vec<T> where std::ops::Range<N>: Iterator, F: Fn(<std::ops::Range<N> as Iterator>::Item) -> T {
    (begin..end).map(|i| func(i)).collect::<Vec::<T>>()
}

/// usizeにキャストするトレイト
pub trait Usize {
    /// usizeにキャストする関数
    fn usize(self) -> usize;
}

/// isizeにキャストするトレイト
pub trait Isize {
    /// isizeにキャストする関数
    fn isize(self) -> isize;
}

/// min関数
#[allow(dead_code)]
pub fn min<T>(left: T, right: T) -> T where T: PartialOrd {
    if left<right {
        left
    } else {
        right
    }
}

/// max関数
#[allow(dead_code)]
pub fn max<T>(left: T, right: T) -> T where T: PartialOrd {
    if left>right {
        left
    } else {
        right
    }
}

/// chminとchmaxのトレイト
pub trait ChminChmax {
    /// challengerのほうが小さければchallengerで上書きする関数
    fn chmin(&mut self, challenger: Self);
    /// challengerのほうが大きければchallengerで上書きする関数
    fn chmax(&mut self, challenger: Self);
}

impl<T> ChminChmax for T where T: Clone + PartialOrd {
    fn chmin(&mut self, challenger: Self) {
        if challenger<*self {
            *self=challenger;
        }
    }
    fn chmax(&mut self, challenger: Self) {
        if challenger>*self {
            *self=challenger;
        }
    }
}

/// 1次元ベクターについてのchminとchmaxのトレイト
pub trait ChminChmaxVec where Self: std::ops::Index<usize> {
    /// ベクターの中身について、添字challengerの値のほうが小さければ上書きする関数
    fn chmin_vec(&mut self, index: usize, challenger: usize);
    /// ベクターの中身について、添字challengerの値のほうが大きければ上書きする関数
    fn chmax_vec(&mut self, index: usize, challenger: usize);
}

impl<T> ChminChmaxVec for Vec<T> where T: Clone + PartialOrd {
    fn chmin_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]<self[index] {
            self[index]=self[challenger].clone();
        }
    }
    fn chmax_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]>self[index] {
            self[index]=self[challenger].clone();
        }
    }
}

impl<T, const N: usize> ChminChmaxVec for [T;N] where T: Clone + PartialOrd {
    fn chmin_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]<self[index] {
            self[index]=self[challenger].clone();
        }
    }
    fn chmax_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]>self[index] {
            self[index]=self[challenger].clone();
        }
    }
}

/// 添字についてchminとchmaxを行うトレイト
pub trait ChminChmaxIndex<T> where T: std::ops::Index<Self>, T::Output: PartialOrd {
    /// vecの現在の添字の値よりchallengerの値のほうが小さければchallengerで上書きする関数
    fn chmin_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが大きければchallengerで上書きする関数
    fn chmax_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが大きくなければchallengerで上書きする関数
    fn chmineq_index(&mut self, vec: &T, challenger: Self);
    /// vecの現在の添字の値よりchallengerの値のほうが小さくなければchallengerで上書きする関数
    fn chmaxeq_index(&mut self, vec: &T, challenger: Self);
}

impl<T> ChminChmaxIndex<T> for usize where T: std::ops::Index<Self>, T::Output: PartialOrd {
    fn chmin_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]<vec[*self] {
            *self=challenger;
        }
    }
    fn chmax_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]>vec[*self] {
            *self=challenger;
        }
    }
    fn chmineq_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]<=vec[*self] {
            *self=challenger;
        }
    }
    fn chmaxeq_index(&mut self, vec: &T, challenger: Self) {
        if vec[challenger]>=vec[*self] {
            *self=challenger;
        }
    }
}

/// 2次元ベクターによるグラフの型
#[allow(dead_code)]
pub type VecGraph=Vec<Vec<(usize,usize)>>;
/// BTreeSetのベクターによるグラフの型
#[allow(dead_code)]
pub type SetGraph=Vec<std::collections::BTreeSet<(usize,usize)>>;

/// グラフについてのトレイト ((usize,usize)の2次元ベクターと(usize,usize)のBTreeSetのベクターについて実装)
pub trait Graph where Self: Sized {
    /// グラフを初期化する関数
    fn new(n: usize) -> Self;
    /// 頂点数を返す関数
    fn size(&self) -> usize;
    /// 辺を追加する関数
    fn push(&mut self, a: usize, b: usize, w: usize);
    /// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
            g.push(b, a, 1);
        }
        g
    }
    /// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_directed_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
        }
        g
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
            g.push(b, a, w);
        }
        g
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
        }
        g
    }
    /// 最短経路の距離を返す関数（is_weightedがtrueでダイクストラ法、falseでBFS）（到達不能ならばusize::MAXが入る）
    fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize>;
    /// グラフからUnion-Find木を構築する関数（0-indexed）
    fn construct_union_find(&self) -> ac_library::Dsu;
    /// グラフが二部グラフであるかを判定し、二部グラフであれば色分けの例を返す関数（返り値の型はOption<Vec<bool>>）
    fn is_bipartite_graph(&self) -> Option<Vec<bool>>;
}

impl Graph for VecGraph {
    fn new(n: usize) -> Self {
        vec![Vec::<(usize,usize)>::new();n]
    }
    fn size(&self) -> usize {
        self.len()
    }
    fn push(&mut self, a: usize, b: usize, w: usize) {
        self[a].push((b,w));
    }
    fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize> {
        let mut dist=vec![usize::MAX;self.size()];
        dist[start]=0;
        if is_weighted {
            let mut pq=RevBinaryHeap::<(usize,usize)>::new();
            pq.push((dist[start],start));
            while let Some((d,v))=pq.pop() {
                if dist[v]<d {
                    continue;
                }
                for &(u,w) in &self[v] {
                    if dist[v]+w<dist[u] {
                        dist[u]=dist[v]+w;
                        pq.push((dist[u],u));
                    }
                }
            }
        } else {
            let mut seen=vec![false;self.size()];
            seen[0]=true;
            let mut queue=std::collections::VecDeque::<usize>::new();
            queue.push_back(0);
            while let Some(v)=queue.pop_front() {
                for &(u,w) in &self[v] {
                    debug_assert_eq!(w, 1);
                    if !seen[u] {
                        dist[u]=dist[v]+w;
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        dist
    }
    fn construct_union_find(&self) -> ac_library::Dsu {
        let mut uf=ac_library::Dsu::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self[v] {
                uf.merge(v, u);
            }
        }
        uf
    }
    fn is_bipartite_graph(&self) -> Option<Vec<bool>> {
        let mut ts=ac_library::TwoSat::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self[v] {
                ts.add_clause(v, true, u, true);
                ts.add_clause(v, false, u, false);
            }
        }
        if ts.satisfiable() {
            Some(ts.answer().to_vec())
        } else {
            None
        }
    }
}

impl Graph for SetGraph {
    fn new(n: usize) -> Self {
        vec![std::collections::BTreeSet::<(usize,usize)>::new();n]
    }
    fn size(&self) -> usize {
        self.len()
    }
    fn push(&mut self, a: usize, b: usize, w: usize) {
        self[a].insert((b,w));
    }
    fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize> {
        let mut dist=vec![usize::MAX;self.size()];
        dist[start]=0;
        if is_weighted {
            let mut pq=RevBinaryHeap::<(usize,usize)>::new();
            pq.push((dist[start],start));
            while let Some((d,v))=pq.pop() {
                if dist[v]<d {
                    continue;
                }
                for &(u,w) in &self[v] {
                    if dist[v]+w<dist[u] {
                        dist[u]=dist[v]+w;
                        pq.push((dist[u],u));
                    }
                }
            }
        } else {
            let mut seen=vec![false;self.size()];
            seen[0]=true;
            let mut queue=std::collections::VecDeque::<usize>::new();
            queue.push_back(0);
            while let Some(v)=queue.pop_front() {
                for &(u,w) in &self[v] {
                    debug_assert_eq!(w, 1);
                    if !seen[u] {
                        dist[u]=dist[v]+w;
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        dist
    }
    fn construct_union_find(&self) -> ac_library::Dsu {
        let mut uf=ac_library::Dsu::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self[v] {
                uf.merge(v, u);
            }
        }
        uf
    }
    fn is_bipartite_graph(&self) -> Option<Vec<bool>> {
        let mut ts=ac_library::TwoSat::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self[v] {
                ts.add_clause(v, true, u, true);
                ts.add_clause(v, false, u, false);
            }
        }
        if ts.satisfiable() {
            Some(ts.answer().to_vec())
        } else {
            None
        }
    }
}

/// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
#[allow(dead_code)]
pub fn construct_graph<G>(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> G where G: Graph {
    G::construct_graph(n, m, ab)
}

/// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
#[allow(dead_code)]
pub fn construct_directed_graph<G>(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> G where G: Graph {
    G::construct_directed_graph(n, m, ab)
}

/// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
#[allow(dead_code)]
pub fn construct_weighted_graph<G>(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> G where G: Graph {
    G::construct_weighted_graph(n, m, abw)
}

/// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
#[allow(dead_code)]
pub fn construct_weighted_directed_graph<G>(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> G where G: Graph {
    G::construct_weighted_directed_graph(n, m, abw)
}

/// 二分探索の関数（整数）
#[allow(dead_code)]
pub fn binary_search<F>(ok: isize, bad: isize, determine: F) -> isize where F: Fn(isize) -> bool {
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
    ok
}

/// 二分探索の関数（浮動小数点数）
#[allow(dead_code)]
pub fn float_binary_search<F>(ok: f64, bad: f64, determine: F, rerror: f64) -> f64 where F: Fn(f64) -> bool {
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
    ok
}

/// 最小値を取り出すことのできる優先度つきキューの構造体
#[allow(dead_code)]
#[derive(Clone, Default, std::fmt::Debug)]
pub struct RevBinaryHeap<T> where T: Ord {
    binary_heap: std::collections::BinaryHeap<std::cmp::Reverse<T>>
}

impl<T> RevBinaryHeap<T> where T: Ord {
    pub fn new() -> Self {
        Self { binary_heap: std::collections::BinaryHeap::<std::cmp::Reverse<T>>::new() }
    }
    pub fn is_empty(&self) -> bool {
        self.binary_heap.is_empty()
    }
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.binary_heap.len()
    }
    pub fn push(&mut self, item: T) {
        self.binary_heap.push(std::cmp::Reverse(item));
    }
    pub fn pop(&mut self) -> Option<T> {
        if !self.is_empty() {
            let std::cmp::Reverse(ret)=self.binary_heap.pop().unwrap();
            Some(ret)
        } else {
            None
        }
    }
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.binary_heap.clear();
    }
}

/// 0(加法単位元)を定義するトレイト
pub trait Zero {
    /// 0(加法単位元)を返す関数
    fn zero_val() -> Self;
}

impl<M> Zero for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn zero_val() -> Self {
        Self::new(0)
    }
}

impl<I> Zero for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn zero_val() -> Self {
        Self::new(0)
    }
}

/// 1(乗法単位元)を定義するトレイト
pub trait One {
    /// 1(乗法単位元)を返す関数
    fn one_val() -> Self;
}

impl<M> One for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn one_val() -> Self {
        Self::new(1)
    }
}

impl<I> One for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn one_val() -> Self {
        Self::new(1)
    }
}

/// ModIntの逆元についてのトレイト
pub trait ModIntInv where Self: Sized {
    /// ModIntの逆元をベクターで列挙する関数（最初の要素には0が入る）
    fn construct_modint_inverses(nmax: usize) -> Vec<Self>;
}

impl<M> ModIntInv for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn construct_modint_inverses(nmax: usize) -> Vec<Self> {
        debug_assert!(M::HINT_VALUE_IS_PRIME);
        let mut invs=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            invs[i]=-invs[Self::modulus() as usize%i]*(Self::modulus() as usize/i);
        }
        invs[0]=Self::new(0);
        invs
    }
}

impl<I> ModIntInv for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn construct_modint_inverses(nmax: usize) -> Vec<Self> {
        let mut invs=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            debug_assert!(Self::modulus() as usize%i > 0);
            invs[i]=-invs[Self::modulus() as usize%i]*(Self::modulus() as usize/i);
        }
        invs[0]=Self::new(0);
        invs
    }
}

/// ModIntの逆元をベクターで列挙する関数（最初の要素には0が入る）
#[allow(dead_code)]
pub fn construct_modint_inverses<M>(nmax: usize) -> Vec<M> where M: ModIntInv {
    ModIntInv::construct_modint_inverses(nmax)
}

/// ModIntの階乗についてのトレイト
pub trait ModIntFact where Self: Sized {
    /// ModIntの階乗をベクターで列挙する関数
    fn construct_modint_facts(nmax: usize) -> Vec<Self>;
    /// ModIntの階乗の逆元をベクターで列挙する関数
    fn construct_modint_fact_inverses(nmax: usize, invs: &Vec<Self>) -> Vec<Self>;
}

impl<M> ModIntFact for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn construct_modint_facts(nmax: usize) -> Vec<Self> {
        let mut facts=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            facts[i]=facts[i-1]*i;
        }
        facts
    }
    fn construct_modint_fact_inverses(nmax: usize, invs: &Vec<Self>) -> Vec<Self> {
        debug_assert!(invs.len() > nmax);
        let mut factinvs=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            factinvs[i]=factinvs[i-1]*invs[i];
        }
        factinvs
    }
}

impl<I> ModIntFact for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn construct_modint_facts(nmax: usize) -> Vec<Self> {
        let mut facts=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            facts[i]=facts[i-1]*i;
        }
        facts
    }
    fn construct_modint_fact_inverses(nmax: usize, invs: &Vec<Self>) -> Vec<Self> {
        debug_assert!(invs.len() > nmax);
        let mut factinvs=vec![Self::new(1);nmax+1];
        for i in 2..=nmax {
            factinvs[i]=factinvs[i-1]*invs[i];
        }
        factinvs
    }
}

/// ModIntの階乗をベクターで列挙する関数
#[allow(dead_code)]
pub fn construct_modint_facts<M>(nmax: usize) -> Vec<M> where M: ModIntFact {
    M::construct_modint_facts(nmax)
}

/// ModIntの階乗の逆元をベクターで列挙する関数
#[allow(dead_code)]
pub fn construct_modint_fact_inverses<M>(nmax: usize, invs: &Vec<M>) -> Vec<M> where M: ModIntFact {
    M::construct_modint_fact_inverses(nmax, invs)
}

/// 累積和についてのトレイト
pub trait PrefixSum {
    /// 累積和のベクターを構築する関数
    fn construct_prefix_sum(&self) -> Self;
    /// 構築した累積和のベクターから部分和を計算する関数（0-indexedの左閉右開区間）
    fn calculate_partial_sum(&self, l: usize, r: usize) -> Self::Output where Self: std::ops::Index<usize>;
}

impl<T> PrefixSum for Vec<T> where T: Clone + Zero + std::ops::Add<Output=T> + std::ops::Sub<Output=T> {
    fn construct_prefix_sum(&self) -> Self {
        let mut prefix_sum=vec![T::zero_val();self.len()+1];
        for i in 0..self.len() {
            prefix_sum[i+1]=prefix_sum[i].clone()+self[i].clone();
        }
        prefix_sum
    }
    fn calculate_partial_sum(&self, l: usize, r: usize) -> <Self as std::ops::Index<usize>>::Output {
        debug_assert!(l < self.len());
        debug_assert!(r <= self.len());
        self[r].clone()-self[l].clone()
    }
}

/// 2次元累積和についてのトレイト
pub trait TwoDimPrefixSum {
    /// 2次元累積和のベクターを構築する関数
    fn construct_2d_prefix_sum(&self) -> Self;
    /// 構築した2次元累積和のベクターから部分和を計算する関数（0-indexedの左閉右開区間）
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <Self::Output as std::ops::Index<usize>>::Output where Self: std::ops::Index<usize>, Self::Output: std::ops::Index<usize>;
}

impl<T> TwoDimPrefixSum for Vec<Vec<T>> where T: Clone + Zero + std::ops::Add<Output=T> + std::ops::Sub<Output=T> {
    fn construct_2d_prefix_sum(&self) -> Self {
        let mut prefix_sum=vec![vec![T::zero_val();self[0].len()+1];self.len()+1];
        for i in 0..self.len() {
            debug_assert_eq!(self[i].len(), self[0].len());
            for j in 0..self[0].len() {
                prefix_sum[i+1][j+1]=prefix_sum[i+1][j].clone()+self[i][j].clone();
            }
        }
        for j in 0..self[0].len() {
            for i in 0..self.len() {
                prefix_sum[i+1][j+1]=prefix_sum[i][j+1].clone()+prefix_sum[i+1][j+1].clone();
            }
        }
        prefix_sum
    }
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <<Self as std::ops::Index<usize>>::Output as std::ops::Index<usize>>::Output {
        debug_assert!(l_i < self.len());
        debug_assert!(l_j < self[0].len());
        debug_assert!(r_i <= self.len());
        debug_assert!(r_j <= self[0].len());
        self[r_i][r_j].clone()-self[r_i][l_j].clone()-self[l_i][r_j].clone()+self[l_i][l_j].clone()
    }
}

/// 素数に関するトレイト
pub trait Primes where Self: Sized {
    /// 素数か判定する関数
    fn is_prime(self) -> bool;
    /// 素数冪か判定する関数
    fn is_prime_power(self) -> bool;
    /// 約数を列挙する関数
    fn enumerate_divisors(self) -> Vec<Self>;
    /// 素因数分解をする関数
    fn prime_factorize(self) -> Vec<(Self,Self)>;
    /// ルジャンドルの定理でselfの階乗がpで何回割り切れるかを計算する関数
    fn legendre_s_formula(self, p: usize) -> usize;
    /// エラトステネスの篩で素数を列挙する関数
    fn sieve_of_eratosthenes(nmax: Self) -> Vec<bool>;
    /// 線形篩で最小素因数を列挙する関数
    fn linear_sieve(nmax: Self) -> (Vec<Self>,Vec<Self>);
    /// 線形篩を用いて素因数分解をする関数
    fn fast_prime_factorize(self, linear_sieve: &Vec<Self>) -> Vec<(Self,Self)>;
}

impl Primes for usize {
    fn is_prime(self) -> bool {
        for i in 2..=num_integer::sqrt(self) {
            if self%i==0 {
                return false;
            }
        }
        true
    }
    fn is_prime_power(mut self) -> bool {
        for i in 2..=num_integer::sqrt(self) {
            if self%i==0 {
                while self%i==0 {
                    self/=i;
                }
                return self==1;
            }
        }
        true
    }
    fn enumerate_divisors(self) -> Vec<Self> {
        let mut divs=Vec::<Self>::new();
        let mut high=Vec::<Self>::new();
        for i in 1..=num_integer::sqrt(self) {
            if self%i==0 {
                divs.push(i);
                if self/i!=i {
                    high.push(self/i);
                }
            }
        }
        while let Some(i)=high.pop() {
            divs.push(i);
        }
        divs
    }
    fn prime_factorize(mut self) -> Vec<(Self,Self)> {
        let mut pes=Vec::<(Self,Self)>::new();
        for i in 2..=num_integer::sqrt(self) {
            if self%i==0 {
                let mut e=0;
                while self%i==0 {
                    e+=1;
                    self/=i;
                }
                pes.push((i,e));
            }
        }
        if self>1 {
            pes.push((self,1));
        }
        pes
    }
    fn legendre_s_formula(mut self, p: usize) -> usize {
        let mut e=0;
        while self>0 {
            e+=self/p;
            self/=p;
        }
        e
    }
    fn sieve_of_eratosthenes(nmax: Self) -> Vec<bool> {
        let mut is_prime=vec![true;nmax+1];
        is_prime[0]=false;
        is_prime[1]=false;
        for i in 2..=nmax {
            if is_prime[i] {
                for j in 2..=nmax/i {
                    is_prime[i*j]=false;
                }
            }
        }
        is_prime
    }
    fn linear_sieve(nmax: Self) -> (Vec<Self>,Vec<Self>) {
        let mut lpf=vec![0;nmax+1];
        let mut primes=Vec::<Self>::new();
        for i in 2..=nmax {
            if lpf[i]==0 {
                lpf[i]=i;
                primes.push(i);
            }
            for &p in &primes {
                if p*i>nmax || p>lpf[i] {
                    break;
                }
                lpf[p*i]=p;
            }
        }
        (lpf, primes)
    }
    fn fast_prime_factorize(mut self, linear_sieve: &Vec<Self>) -> Vec<(Self,Self)> {
        debug_assert!(linear_sieve.len() > self);
        let mut pes=Vec::<(Self,Self)>::new();
        let mut p=linear_sieve[self];
        let mut e=0;
        while self>1 {
            if p==linear_sieve[self] {
                e+=1;
            } else {
                pes.push((p,e));
                p=linear_sieve[self];
                e=1;
            }
            self/=linear_sieve[self];
        }
        pes.push((p,e));
        pes
    }
}

/// エラトステネスの篩で素数を列挙する関数
#[allow(dead_code)]
pub fn sieve_of_eratosthenes<T>(nmax: T) -> Vec<bool> where T: Primes {
    T::sieve_of_eratosthenes(nmax)
}

/// 線形篩で最小素因数を列挙する関数
#[allow(dead_code)]
pub fn linear_sieve<T>(nmax: T) -> (Vec<T>,Vec<T>) where T: Primes {
    T::linear_sieve(nmax)
}

/// 2つ以上の数の最大公約数を返すマクロ
#[macro_export]
#[allow(unused_macros)]
macro_rules! gcd {
    ($l:expr,$r:expr) => {
        num_integer::gcd($l,$r)
    };
    ($l:expr,$r:expr,$($vars:expr),+) => {
        num_integer::gcd($l,gcd!($r,$($vars),+))
    };
}

/// 配列やベクターの中身の最大公約数をとるトレイト
pub trait VecGCD where Self: std::ops::Index<usize> {
    /// 配列やベクターの中身の最大公約数を返す関数
    fn gcd(&self) -> Self::Output;
}

impl<T> VecGCD for Vec<T> where T: Copy + num::Zero + num_integer::Integer {
    fn gcd(&self) -> T {
        let mut gcd=num::zero();
        for &var in self {
            gcd=num_integer::gcd(gcd, var);
        }
        gcd
    }
}

impl<T> VecGCD for [T] where T: Copy + num::Zero + num_integer::Integer {
    fn gcd(&self) -> T {
        let mut gcd=num::zero();
        for &var in self {
            gcd=num_integer::gcd(gcd, var);
        }
        gcd
    }
}

impl<T, const N: usize> VecGCD for [T;N] where T: Copy + num::Zero + num_integer::Integer {
    fn gcd(&self) -> T {
        let mut gcd=num::zero();
        for &var in self {
            gcd=num_integer::gcd(gcd, var);
        }
        gcd
    }
}

/// 2つ以上の数の最小公倍数を返すマクロ
#[macro_export]
#[allow(unused_macros)]
macro_rules! lcm {
    ($l:expr,$r:expr) => {
        num_integer::lcm($l,$r)
    };
    ($l:expr,$r:expr,$($vars:expr),+) => {
        num_integer::lcm($l,lcm!($r,$($vars),+))
    };
}

/// 配列やベクターの中身の最小公倍数をとるトレイト
pub trait VecLCM where Self: std::ops::Index<usize> {
    /// 配列やベクターの中身の最小公倍数を返す関数
    fn lcm(&self) -> Self::Output;
}

impl<T> VecLCM for Vec<T> where T: Copy + num::One + num_integer::Integer {
    fn lcm(&self) -> T {
        let mut lcm=num::one();
        for &var in self {
            lcm=num_integer::lcm(lcm, var);
        }
        lcm
    }
}

impl<T> VecLCM for [T] where T: Copy + num::One + num_integer::Integer {
    fn lcm(&self) -> T {
        let mut lcm=num::one();
        for &var in self {
            lcm=num_integer::lcm(lcm, var);
        }
        lcm
    }
}

impl<T, const N: usize> VecLCM for [T;N] where T: Copy + num::One + num_integer::Integer {
    fn lcm(&self) -> T {
        let mut lcm=num::one();
        for &var in self {
            lcm=num_integer::lcm(lcm, var);
        }
        lcm
    }
}

/// N1×N2行列の構造体（num::powで行列累乗を計算できる）
#[derive(Clone, std::fmt::Debug)]
pub struct Matrix<T, const N1: usize, const N2: usize> {
    matrix: [[T;N2];N1]
}

impl<T, const N1: usize, const N2: usize> std::ops::Deref for Matrix<T,N1,N2> {
    type Target = [[T;N2];N1];
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::DerefMut for Matrix<T,N1,N2> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.matrix
    }
}

impl<T, const N1: usize, const N2: usize> PartialEq for Matrix<T,N1,N2> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        let mut ret=true;
        for i in 0..N1 {
            for j in 0..N2 {
                if self[i][j]!=other[i][j] {
                    ret=false;
                }
            }
        }
        ret
    }
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl<T, const N1: usize, const N2: usize> Zero for Matrix<T,N1,N2> where T: Copy + Zero {
    fn zero_val() -> Self {
        Self { matrix: [[T::zero_val();N2];N1] }
    }
}

impl<T, const N1: usize, const N2: usize> num::Zero for Matrix<T,N1,N2> where T: Copy + Zero + std::ops::Add<Output=T> + PartialEq {
    fn zero() -> Self {
        Self::zero_val()
    }
    fn is_zero(&self) -> bool {
        *self==Self::zero()
    }
}

impl<T, const N: usize> One for Matrix<T,N,N> where T: Copy + Zero + One {
    fn one_val() -> Self {
        let mut matrix=Self::zero_val();
        for i in 0..N {
            matrix[i][i]=T::one_val();
        }
        matrix
    }
}

impl<T, const N: usize> num::One for Matrix<T,N,N> where T: Copy + Zero + One + std::ops::AddAssign + std::ops::Mul<Output=T> + PartialEq {
    fn one() -> Self {
        Self::one_val()
    }
    fn is_one(&self) -> bool {
        *self==Self::one()
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::AddAssign for Matrix<T,N1,N2> where T: Clone + std::ops::AddAssign {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N1 {
            for j in 0..N2 {
                self[i][j]+=rhs[i][j].clone();
            }
        }
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::Add for Matrix<T,N1,N2> where T: Clone + std::ops::Add<Output=T> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N1 {
            for j in 0..N2 {
                self[i][j]=self[i][j].clone()+rhs[i][j].clone();
            }
        }
        self
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::Neg for Matrix<T,N1,N2> where T: Clone + std::ops::Neg<Output=T> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for i in 0..N1 {
            for j in 0..N2 {
                self[i][j]=-self[i][j].clone();
            }
        }
        self
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::SubAssign for Matrix<T,N1,N2> where T: Clone + std::ops::SubAssign {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N1 {
            for j in 0..N2 {
                self[i][j]-=rhs[i][j].clone();
            }
        }
    }
}

impl<T, const N1: usize, const N2: usize> std::ops::Sub for Matrix<T,N1,N2> where T: Clone + std::ops::Sub<Output=T> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N1 {
            for j in 0..N2 {
                self[i][j]=self[i][j].clone()-rhs[i][j].clone();
            }
        }
        self
    }
}

impl<T, const N: usize> std::ops::MulAssign for Matrix<T,N,N> where T: Copy + Zero + std::ops::AddAssign + std::ops::Mul<Output=T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self=self.clone()*rhs;
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize> std::ops::Mul<Matrix<T,N2,N3>> for Matrix<T,N1,N2> where T: Copy + Zero + std::ops::AddAssign + std::ops::Mul<Output=T> {
    type Output = Matrix<T,N1,N3>;
    fn mul(self, rhs: Matrix<T,N2,N3>) -> Self::Output {
        let mut prod=Self::Output::zero_val();
        for i in 0..N1 {
            for j in 0..N3 {
                for k in 0..N2 {
                    prod[i][j]+=self[i][k].clone()*rhs[k][j].clone();
                }
            }
        }
        prod
    }
}

/// FunctionalGraphのトレイト
pub trait FunctionalGraph where Self: Sized {
    /// FunctionalGraphのダブリングを前計算する関数
    fn doubling(&self, cntmax: usize) -> Vec<Self>;
}

impl FunctionalGraph for Vec<usize> {
    fn doubling(&self, cntmax: usize) -> Vec<Self> {
        let log=cntmax.ilog2() as usize+1;
        let mut doubling=vec![vec![0;log];self.len()];
        for i in 0..self.len() {
            doubling[i][0]=self[i];
        }
        for k in 0..log-1 {
            for i in 0..self.len() {
                doubling[i][k+1]=doubling[doubling[i][k]][k];
            }
        }
        doubling
    }
}

/// ダブリングのトレイト
pub trait Doubling where Self: Sized + std::ops::Index<usize>, Self::Output: std::ops::Index<usize> {
    /// FunctionalGraphでstartからcnt回移動した先を返す関数（0-indexed）
    fn terminus(&self, start: usize, cnt: usize) -> <Self::Output as std::ops::Index<usize>>::Output;
}

impl Doubling for Vec<Vec<usize>> {
    fn terminus(&self, mut start: usize, mut cnt: usize) -> usize {
        let mut k=0;
        while cnt>0 {
            if (cnt&1)>0 {
                start=self[start][k];
            }
            cnt/=2;
            k+=1;
        }
        start
    }
}

/// NTT素数のベクターで形式的冪級数を扱うトレイト
pub trait FPS {
    /// 形式的冪級数の和を割り当てる関数
    fn fps_add_assign(&mut self, g: &Self);
    /// 形式的冪級数の和を返す関数
    fn add(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の差を割り当てる関数
    fn fps_sub_assign(&mut self, g: &Self);
    /// 形式的冪級数の差を返す関数
    fn sub(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の定数倍を割り当てる関数
    fn fps_scalar_assign(&mut self, k: isize);
    /// 形式的冪級数の定数倍を返す関数
    fn scalar(f: &Self, k: isize) -> Self;
    /// 形式的冪級数の積を割り当てる関数
    fn fps_mul_assign(&mut self, g: &Self);
    /// 形式的冪級数の積を返す関数
    fn mul(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の逆元を返す関数
    fn fps_inv(&self) -> Self;
    /// 形式的冪級数の商を割り当てる関数
    fn fps_div_assign(&mut self, g: &Self);
    /// 形式的冪級数の商を返す関数
    fn div(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の導関数を割り当てる関数
    fn fps_diff_assign(&mut self);
    /// 形式的冪級数の導関数を返す関数
    fn fps_diff(&self) -> Self;
    /// 形式的冪級数の定積分を割り当てる関数
    fn fps_int_assign(&mut self);
    /// 形式的冪級数の定積分を返す関数
    fn fps_int(&self) -> Self;
    /// 形式的冪級数の対数を割り当てる関数
    fn fps_log_assign(&mut self);
    /// 形式的冪級数の対数を返す関数
    fn fps_log(&self) -> Self;
    /// 形式的冪級数の指数を割り当てる関数
    fn fps_exp_assign(&mut self);
    /// 形式的冪級数の指数を返す関数
    fn fps_exp(&self) -> Self;
    /// 形式的冪級数の冪を割り当てる関数
    fn fps_pow_assign(&mut self, k: usize);
    /// 形式的冪級数の冪を返す関数
    fn fps_pow(&self, k: usize) -> Self;
}

impl<M> FPS for Vec<ac_library::StaticModInt<M>> where M: ac_library::Modulus {
    fn fps_add_assign(&mut self, g: &Self) {
        debug_assert_eq!(self.len(), g.len());
        let n=self.len()-1;
        for i in 0..=n {
            self[i]+=g[i];
        }
    }
    fn add(f: &Self, g: &Self) -> Self {
        debug_assert_eq!(f.len(), g.len());
        let mut h=f.clone();
        h.fps_add_assign(&g);
        h
    }
    fn fps_sub_assign(&mut self, g: &Self) {
        debug_assert_eq!(self.len(), g.len());
        let n=self.len()-1;
        for i in 0..=n {
            self[i]-=g[i];
        }
    }
    fn sub(f: &Self, g: &Self) -> Self {
        debug_assert_eq!(f.len(), g.len());
        let mut h=f.clone();
        h.fps_sub_assign(&g);
        h
    }
    fn fps_scalar_assign(&mut self, k: isize) {
        let n=self.len()-1;
        for i in 0..=n {
            self[i]*=k;
        }
    }
    fn scalar(f: &Self, k: isize) -> Self {
        let mut h=f.clone();
        h.fps_scalar_assign(k);
        h
    }
    fn fps_mul_assign(&mut self, g: &Self) {
        debug_assert_eq!(self.len(), g.len());
        let n=self.len()-1;
        let h=FPS::mul(self, g);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn mul(f: &Self, g: &Self) -> Self {
        debug_assert_eq!(f.len(), g.len());
        let n=f.len()-1;
        ac_library::convolution::convolution(&f[0..=n], &g[0..=n])[0..=n].to_vec()
    }
    fn fps_inv(&self) -> Self {
        let n=self.len()-1;
        let mut inv=vec![ac_library::StaticModInt::<M>::new(0);n+1];
        inv[0]=self[0].inv();
        let mut curdeg=1;
        while curdeg<=n {
            curdeg*=2;
            let mut f=self[0..std::cmp::min(curdeg,n+1)].to_vec();
            let mut g=vec![ac_library::StaticModInt::<M>::new(0);std::cmp::min(curdeg,n+1)];
            for i in 0..curdeg/2 {
                g[i]=inv[i];
            }
            f.fps_mul_assign(&g);
            f.fps_mul_assign(&g);
            for i in curdeg/2..std::cmp::min(curdeg,n+1) {
                inv[i]-=f[i];
            }
        }
        inv
    }
    fn fps_div_assign(&mut self, g: &Self) {
        debug_assert_eq!(self.len(), g.len());
        self.fps_mul_assign(&g.fps_inv());
    }
    fn div(f: &Self, g: &Self) -> Self {
        debug_assert_eq!(f.len(), g.len());
        let mut h=f.clone();
        h.fps_div_assign(&g);
        h
    }
    fn fps_diff_assign(&mut self) {
        let n=self.len()-1;
        for i in 0..n {
            self[i]=self[i+1]*(i+1);
        }
    }
    fn fps_diff(&self) -> Self {
        let mut h=self.clone();
        h.fps_diff_assign();
        h
    }
    fn fps_int_assign(&mut self) {
        let n=self.len()-1;
        for i in (1..=n).rev() {
            self[i]=self[i-1]/i;
        }
        self[0]=ac_library::StaticModInt::<M>::new(0);
    }
    fn fps_int(&self) -> Self {
        let mut h=self.clone();
        h.fps_int_assign();
        h
    }
    fn fps_log_assign(&mut self) {
        let n=self.len()-1;
        let h=FPS::fps_log(self);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn fps_log(&self) -> Self {
        let mut h=self.clone();
        h.fps_diff_assign();
        h.fps_div_assign(self);
        h.fps_int_assign();
        h
    }
    fn fps_exp_assign(&mut self) {
        let n=self.len()-1;
        let h=FPS::fps_exp(self);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn fps_exp(&self) -> Self {
        debug_assert_eq!(self[0], ac_library::StaticModInt::<M>::new(0));
        let n=self.len()-1;
        let mut exp=vec![ac_library::StaticModInt::<M>::new(0);n+1];
        exp[0]=ac_library::StaticModInt::<M>::new(1);
        let mut curdeg=1;
        while curdeg<=n {
            curdeg*=2;
            let mut fc=vec![ac_library::StaticModInt::<M>::new(0);n+1];
            for i in 0..std::cmp::min(curdeg,n+1) {
                fc[i]=self[i];
            }
            fc.fps_sub_assign(&FPS::fps_log(&exp));
            fc[0]+=1;
            exp.fps_mul_assign(&fc);
        }
        exp
    }
    fn fps_pow_assign(&mut self, k: usize) {
        let n=self.len()-1;
        let mut lower=(ac_library::StaticModInt::<M>::new(1),0);
        for i in 0..=n {
            if self[i]!=ac_library::StaticModInt::<M>::new(0) {
                lower=(self[i],i);
                break;
            }
        }
        for i in 0..=n-lower.1 {
            self[i]=self[i+lower.1]/lower.0;
        }
        for i in n-lower.1+1..=n {
            self[i]=ac_library::StaticModInt::<M>::new(0);
        }
        self.fps_log_assign();
        self.fps_scalar_assign(k as isize);
        self.fps_exp_assign();
        for i in (lower.1*k..=n).rev() {
            self[i]=self[i-lower.1*k]*lower.0.pow(k as u64);
        }
        for i in 0..std::cmp::min(lower.1*k,n+1) {
            self[i]=ac_library::StaticModInt::<M>::new(0);
        }
    }
    fn fps_pow(&self, k: usize) -> Self {
        let mut h=self.clone();
        h.fps_pow_assign(k);
        h
    }
}

/// 高速ゼータ変換および高速メビウス変換についてのトレイト
pub trait ZetaMobius {
    /// 上位集合についての高速ゼータ変換をする関数
    fn zeta_superset_transform(&mut self);
    /// 上位集合についての高速メビウス変換をする関数
    fn mobius_superset_transform(&mut self);
    /// 部分集合についての高速ゼータ変換をする関数
    fn zeta_subset_transform(&mut self);
    /// 部分集合についての高速メビウス変換をする関数
    fn mobius_subset_transform(&mut self);
}

impl<T> ZetaMobius for Vec<T> where T: Clone + std::ops::Add<Output=T> + std::ops::Sub<Output=T> {
    fn zeta_superset_transform(&mut self) {
        let n=self.len();
        let mut i=1;
        while i<n {
            for j in 0..n {
                if j&i==0 {
                    self[j]=self[j].clone()+self[j|i].clone();
                }
            }
            i<<=1;
        }
    }
    fn mobius_superset_transform(&mut self) {
        let n=self.len();
        let mut i=1;
        while i<n {
            for j in 0..n {
                if j&i==0 {
                    self[j]=self[j].clone()-self[j|i].clone();
                }
            }
            i<<=1;
        }
    }
    fn zeta_subset_transform(&mut self) {
        let n=self.len();
        let mut i=1;
        while i<n {
            for j in 0..n {
                if j&i==0 {
                    self[j|i]=self[j|i].clone()+self[j].clone();
                }
            }
            i<<=1;
        }
    }
    fn mobius_subset_transform(&mut self) {
        let n=self.len();
        let mut i=1;
        while i<n {
            for j in 0..n {
                if j&i==0 {
                    self[j|i]=self[j|i].clone()-self[j].clone();
                }
            }
            i<<=1;
        }
    }
}

/// 単一の文字を数値に変換する関数のトレイト
pub trait FromChar {
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
        self.from_foo('0', '9')
    }
    fn from_uppercase(self) -> usize {
        self.from_foo('A', 'Z')
    }
    fn from_lowercase(self) -> usize {
        self.from_foo('a', 'z')
    }
}

/// 数値を単一の文字に変換する関数のトレイト
pub trait ToChar {
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
        self.to_foo('0', 10)
    }
    fn to_uppercase(self) -> char {
        self.to_foo('A', 26)
    }
    fn to_lowercase(self) -> char {
        self.to_foo('a', 26)
    }
}

/// 数の文字列の10進法への変換についてのトレイト
pub trait ToDecimal {
    /// radix進法の数の文字列を10進数の数値へ変換する関数
    fn to_decimal(&self, radix: usize) -> usize;
}

impl ToDecimal for String {
    fn to_decimal(&self, radix: usize) -> usize {
        usize::from_str_radix(&self, radix as u32).unwrap()
    }
}

/// 2進法での桁数を求めるトレイト
pub trait BitDigits {
    /// 2進法での桁数を求める関数
    fn bit_digits(self) -> usize;
}

/// 10のi乗のstatic定数
#[allow(dead_code)]
pub static E: [usize;19]=gen_e();

/// 10のi乗のstatic定数を生成するconst関数
const fn gen_e() -> [usize;19] {
    let mut e=[1;19];
    let mut i=1;
    while i<19 {
        e[i]=e[i-1]*10;
        i+=1;
    }
    e
}

/// 素数位数の有限体の元で表された有理数を推測するトレイト
pub trait RationalReconstruct {
    /// 関数の返り値の型
    type Output;
    /// 素数位数の有限体の元で表された有理数を推測する関数
    fn rational_reconstruct(&self) -> Self::Output;
}

impl<M> RationalReconstruct for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    type Output = f64;
    fn rational_reconstruct(&self) -> Self::Output {
        let mut v=(M::VALUE as isize, 0);
        let mut w=(self.val() as isize, 1);
        while w.0*w.0*2>M::VALUE as isize {
            let q=v.0/w.0;
            let z=(v.0-q*w.0, v.1-q*w.1);
            v=w;
            w=z;
        }
        w.0 as f64 / w.1 as f64
    }
}

impl<I> RationalReconstruct for ac_library::DynamicModInt<I> where I: ac_library::Id {
    type Output = f64;
    fn rational_reconstruct(&self) -> Self::Output {
        let mut v=(Self::modulus() as isize, 0);
        let mut w=(self.val() as isize, 1);
        while w.0*w.0*2>Self::modulus() as isize {
            let q=v.0/w.0;
            let z=(v.0-q*w.0, v.1-q*w.1);
            v=w;
            w=z;
        }
        w.0 as f64 / w.1 as f64
    }
}

impl<T> RationalReconstruct for Vec<T> where T: RationalReconstruct {
    type Output = Vec<T::Output>;
    fn rational_reconstruct(&self) -> Self::Output {
        self.iter().map(|v| v.rational_reconstruct()).collect::<Self::Output>()
    }
}

/// プリミティブな整数型についてimplを定義するマクロ
macro_rules! impl_integer {
    ($($ty:ty),*) => {
        $(
            impl Usize for $ty {
                fn usize(self) -> usize {
                    self as usize
                }
            }

            impl Isize for $ty {
                fn isize(self) -> isize {
                    self as isize
                }
            }

            impl Zero for $ty {
                fn zero_val() -> Self {
                    0
                }
            }

            impl One for $ty {
                fn one_val() -> Self {
                    1
                }
            }

            impl BitDigits for $ty {
                fn bit_digits(self) -> usize {
                    self.ilog2() as usize
                }
            }
        )*
    }
}

impl_integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

/// 結合テスト用（AtCoderの環境に含まれないクレートを使用していることに注意）
#[macro_export]
#[allow(unused_macros)]
macro_rules! tests {
    ($name:expr,$($num:expr,$input:expr,$output:expr),*) => {
        $(
            paste::paste! {
                #[test]
                fn [<$name _sample_ $num>]() {
                    let mut cmd=assert_cmd::Command::cargo_bin($name).unwrap();
                    cmd.write_stdin($input);
                    cmd.assert().success().stdout(predicates::prelude::predicate::str::diff($output));
                }
            }
        )*
    };
}

// not_leonian_ac_lib until this line
