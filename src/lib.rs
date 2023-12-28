//! <https://github.com/NotLeonian/not_leonian_ac_lib>  
//!   
//! Copyright (c) 2023 Not_Leonian  
//! Released under the MIT license  
//! <https://opensource.org/licenses/mit-license.php>  

/// 1つ以上の値を1行で出力するマクロ
#[macro_export]
macro_rules! outputln {
    ($var:expr) => {
        println!("{}",$var)
    };
    ($var:expr,$($vars:expr),+) => {
        print!("{} ",$var);
        outputln!($($vars),+);
    };
}

/// 1つ以上の値のDebugを1行で出力するマクロ
#[macro_export]
macro_rules! debugln {
    ($var:expr) => {
        println!("{:?}",$var)
    };
    ($var:expr,$($vars:expr),+) => {
        print!("{:?} ",$var);
        debugln!($($vars),+);
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
pub fn outputif<T1,T2>(cond: bool, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display {
    if cond {
        println!("{}",ok);
    } else {
        println!("{}",bad);
    }
}

/// 条件によって"Yes"または"No"の1行を出力する関数
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
macro_rules! eoutputln {
    ($var:expr) => {
        #[cfg(debug_assertions)]
        eprintln!("{}",$var)
    };
    ($var:expr,$($vars:expr),+) => {
        #[cfg(debug_assertions)]
        eprint!("{} ",$var);
        eoutputln!($($vars),+);
    };
}

/// 1つ以上の値のDebugを1行でstderrに出力するマクロ
#[macro_export]
macro_rules! edebugln {
    ($var:expr) => {
        #[cfg(debug_assertions)]
        eprintln!("{:?}",$var)
    };
    ($var:expr,$($vars:expr),+) => {
        #[cfg(debug_assertions)]
        eprint!("{:?} ",$var);
        edebugln!($($vars),+);
    };
}

/// 配列やベクターの中身を1行でstderrに出力するトレイト
pub trait Eoutputln {
    /// 配列やベクターの中身を1行でstderrに出力する関数
    fn eoutputln(&self);
}

impl<T> Eoutputln for Vec<T> where T: std::fmt::Display {
    fn eoutputln(&self) {
        #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
        for v in self {
            v.eoutputln();
        }
    }
}

impl<T> Eoutputlns for [T] where T: Eoutputln {
    fn eoutputlns(&self) {
        #[cfg(debug_assertions)]
        for v in self {
            v.eoutputln();
        }
    }
}

impl<T, const N: usize> Eoutputlns for [T;N] where T: Sized + Eoutputln {
    fn eoutputlns(&self) {
        #[cfg(debug_assertions)]
        for v in self {
            v.eoutputln();
        }
    }
}

/// 条件によって変わる1行をstderrに出力する関数（引数は順に条件と真の場合、偽の場合の出力）
#[allow(unused_variables)]
pub fn eoutputif<T1,T2>(cond: bool, ok: T1, bad: T2) where T1: std::fmt::Display, T2: std::fmt::Display {
    #[cfg(debug_assertions)]
    if cond {
        eprintln!("{}",ok);
    } else {
        eprintln!("{}",bad);
    }
}

/// 条件によって"Yes"または"No"の1行をstderrに出力する関数
#[allow(unused_variables)]
pub fn eoutput_yes_or_no(cond: bool) {
    #[cfg(debug_assertions)]
    eoutputif(cond, "Yes", "No");
}

/// 存在すれば値を、存在しなければ-1をstderrに出力するトレイト
pub trait EoutputValOr {
    /// 値がmaxより小さければ自身を出力し、maxであれば-1をstderrに出力する関数
    fn eoutput_val_or(self, max: Self);
}

impl EoutputValOr for usize {
    #[allow(unused_variables)]
    fn eoutput_val_or(self, max: Self) {
        #[cfg(debug_assertions)]
        if self<max {
            eprintln!("{}",self);
        } else {
            eprintln!("-1");
        }
    }
}

/// 配列やベクターに末尾から数えたインデックスでアクセスするトレイト
pub trait GetFromLast {
    /// 配列やベクターに末尾から数えたインデックスでアクセスする関数（1-indexedであることに注意）
    fn get_from_last(&self, i: usize) -> &Self::Output where Self: std::ops::Index<usize>;
}

impl<T> GetFromLast for Vec<T> {
    fn get_from_last(&self, i: usize) -> &<Vec<T> as std::ops::Index<usize>>::Output {
        &self[self.len()-i]
    }
}

impl<T> GetFromLast for [T] {
    fn get_from_last(&self, i: usize) -> &<[T] as std::ops::Index<usize>>::Output {
        &self[self.len()-i]
    }
}

impl<T, const N: usize> GetFromLast for [T;N] {
    fn get_from_last(&self, i: usize) -> &<[T;N] as std::ops::Index<usize>>::Output {
        &self[self.len()-i]
    }
}

impl<T> GetFromLast for std::collections::VecDeque<T> {
    fn get_from_last(&self, i: usize) -> &<std::collections::VecDeque<T> as std::ops::Index<usize>>::Output {
        &self[self.len()-i]
    }
}

/// 配列やベクターに末尾から数えたインデックスでmutでアクセスするトレイト
pub trait GetMutFromLast {
    /// 配列やベクターに末尾から数えたインデックスでmutでアクセスする関数（1-indexedであることに注意）
    fn get_mut_from_last(&mut self, i: usize) -> &mut Self::Output where Self: std::ops::Index<usize>;
}

impl<T> GetMutFromLast for Vec<T> {
    fn get_mut_from_last(&mut self, i: usize) -> &mut <Vec<T> as std::ops::Index<usize>>::Output {
        let len=self.len();
        &mut self[len-i]
    }
}

impl<T> GetMutFromLast for [T] {
    fn get_mut_from_last(&mut self, i: usize) -> &mut <[T] as std::ops::Index<usize>>::Output {
        let len=self.len();
        &mut self[len-i]
    }
}

impl<T, const N: usize> GetMutFromLast for [T;N] {
    fn get_mut_from_last(&mut self, i: usize) -> &mut <[T;N] as std::ops::Index<usize>>::Output {
        let len=self.len();
        &mut self[len-i]
    }
}

impl<T> GetMutFromLast for std::collections::VecDeque<T> {
    fn get_mut_from_last(&mut self, i: usize) -> &mut <std::collections::VecDeque<T> as std::ops::Index<usize>>::Output {
        let len=self.len();
        &mut self[len-i]
    }
}

/// for文風にbeginからendまでの結果を格納したベクターを生成する関数（0-indexedの左閉右開区間）
pub fn vec_range<N,F,T>(begin: N, end: N, func: F) -> Vec<T> where std::ops::Range<N>: Iterator, F: Fn(<std::ops::Range<N> as Iterator>::Item) -> T {
    (begin..end).map(|i| func(i)).collect::<Vec::<T>>()
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

/// ベクターをソートして重複している要素を取り除く関数のトレイト
pub trait SortAndDedup {
    /// ベクターをソートして重複している要素を取り除く関数
    fn sort_and_dedup(&mut self);
}

impl<T> SortAndDedup for Vec<T> where T: Ord {
    fn sort_and_dedup(&mut self) {
        self.sort();
        self.dedup();
    }
}

/// 配列やベクターの要素の総和のトレイト
pub trait Sum<'a> {
    /// 返り値の型
    type T;
    /// 配列やベクターの要素の総和を返す関数
    fn sum(&'a self) -> Self::T;
}

impl<'a, T> Sum<'a> for Vec<T> where T: 'a + std::iter::Sum<&'a T> {
    type T = T;
    fn sum(&'a self) -> T {
        <T as std::iter::Sum<&'a T>>::sum(self.iter())
    }
}

impl<'a, T> Sum<'a> for [T] where T: 'a + std::iter::Sum<&'a T> {
    type T = T;
    fn sum(&'a self) -> T {
        <T as std::iter::Sum<&'a T>>::sum(self.iter())
    }
}

impl<'a, T, const N: usize> Sum<'a> for [T;N] where T: 'a + std::iter::Sum<&'a T> {
    type T = T;
    fn sum(&'a self) -> T {
        <T as std::iter::Sum<&'a T>>::sum(self.iter())
    }
}

/// 配列やベクターの要素の総積のトレイト
pub trait Product<'a> {
    /// 返り値の型
    type T;
    /// 配列やベクターの要素の総積を返す関数
    fn product(&'a self) -> Self::T;
}

impl<'a, T> Product<'a> for Vec<T> where T: 'a + std::iter::Product<&'a T> {
    type T = T;
    fn product(&'a self) -> T {
        <T as std::iter::Product<&'a T>>::product(self.iter())
    }
}

impl<'a, T> Product<'a> for [T] where T: 'a + std::iter::Product<&'a T> {
    type T = T;
    fn product(&'a self) -> T {
        <T as std::iter::Product<&'a T>>::product(self.iter())
    }
}

impl<'a, T, const N: usize> Product<'a> for [T;N] where T: 'a + std::iter::Product<&'a T> {
    type T = T;
    fn product(&'a self) -> T {
        <T as std::iter::Product<&'a T>>::product(self.iter())
    }
}

/// min関数
pub fn min<T>(left: T, right: T) -> T where T: PartialOrd {
    if left<right {
        left
    } else {
        right
    }
}

/// max関数
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

/// ソートされているベクターどうしを、ソートされた1つのベクターへマージする関数
pub fn merge_vecs<T>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> where T: Clone + PartialOrd {
    itertools::Itertools::merge(a.iter(), b.iter()).cloned().collect()
}

/// do-while文のマクロ
#[macro_export]
macro_rules! do_while {
    ($cond:expr,$block:block) => {
        loop {
            $block
            if !$cond {
                break;
            }
        }
    };
}

/// ModInt998244353を表す型
pub type Mint=ac_library::ModInt998244353;

/// ModInt1000000007を表す型
pub type OldMint=ac_library::ModInt1000000007;

/// DFSやBFSのイテレータのもつ列挙型（Vertexが行きであるかどうかと頂点番号、VertexEdgeWeightが頂点番号と隣接する頂点番号とその辺の重み）
pub enum GraphSearch {
    Vertex(bool,usize),
    VertexEdgeWeight(usize,usize,usize)
}

/// 2次元ベクターによるグラフの構造体
#[derive(Clone, Default, Debug)]
pub struct VecGraph {
    graph: Vec<Vec<(usize,usize)>>
}

impl VecGraph {
    /// グラフを初期化する関数
    pub fn new(n: usize) -> Self {
        Self { graph: vec![Vec::<(usize,usize)>::new();n] }
    }
    /// 頂点数を返す関数
    pub fn size(&self) -> usize {
        self.graph.len()
    }
    /// 隣接リストの参照を返す関数
    pub fn get(&self) -> &Vec<Vec<(usize,usize)>> {
        &self.graph
    }
    /// 隣接リストの可変参照を返す関数
    pub fn get_mut(&mut self) -> &mut Vec<Vec<(usize,usize)>> {
        &mut self.graph
    }
    /// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g=VecGraph::new(n);
        for &(a,b) in ab {
            g.graph[a].push((b, 1));
            g.graph[b].push((a, 1));
        }
        g
    }
    /// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_directed_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g=VecGraph::new(n);
        for &(a,b) in ab {
            g.graph[a].push((b, 1));
        }
        g
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=VecGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, w));
            g.graph[b].push((a, w));
        }
        g
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=VecGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, w));
        }
        g
    }
    /// DFSのイテレータを返す関数
    pub fn dfs(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=Vec::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some(&(u,w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=stack.pop() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 全ての辺を通るDFSのイテレータを返す関数
    pub fn dfs_all_edges(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=Vec::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some(&(u,w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u);
                    }
                    return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                }
                if let Some(v)=stack.pop() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 帰りにも頂点を訪れるDFSのイテレータを返す関数
    pub fn dfs_postorder(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=vec![start+self.size()];
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some(&(u,w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u+self.size());
                        stack.push(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=stack.pop() {
                    if v<self.size() {
                        vertex=v;
                        it=self.graph[v].iter();
                        Some(GraphSearch::Vertex(true, v))
                    } else {
                        Some(GraphSearch::Vertex(false, v-self.size()))
                    }
                } else {
                    None
                }
            }
        })
    }
    /// BFSのイテレータを返す関数
    pub fn bfs(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut queue=std::collections::VecDeque::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some(&(u,w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        queue.push_front(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=queue.pop_back() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 最短経路の距離を返す関数（is_weightedがtrueでダイクストラ法、falseでBFS）（到達不能ならばusize::MAXが入る）
    pub fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize> {
        let mut dist=vec![usize::MAX;self.size()];
        dist[start]=0;
        if is_weighted {
            let mut pq=RevBinaryHeap::<(usize,usize)>::new();
            pq.push((dist[start],start));
            while let Some((d,v))=pq.pop() {
                if dist[v]<d {
                    continue;
                }
                for &(u,w) in &self.graph[v] {
                    if dist[v].saturating_add(w)<dist[u] {
                        dist[u]=dist[v].saturating_add(w);
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
                for &(u,w) in &self.graph[v] {
                    debug_assert_eq!(w, 1);
                    if !seen[u] {
                        dist[u]=dist[v].saturating_add(w);
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        dist
    }
    /// ワーシャル・フロイド法の関数（到達不能ならばusize::MAXが入る）
    pub fn floyd_warshall(&self) -> Vec<Vec<usize>> {
        let mut dist=vec![vec![usize::MAX;self.size()];self.size()];
        for v in 0..self.size() {
            for &(u,w) in &self.graph[v] {
                dist[v][u]=w;
                dist[u][v]=w;
            }
        }
        for m in 0..self.size() {
            for v in 0..self.size() {
                for u in 0..self.size() {
                    dist[v][u]=min(dist[v][m].saturating_add(dist[m][u]), dist[v][u]);
                }
            }
        }
        dist
    }
    /// グラフに閉路がなければ、頂点をトポロジカルソートした結果を返す関数（返り値はOption）
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut ret=Vec::<usize>::new();
        let mut indeg=vec![0;self.size()];
        for v in 0..self.size() {
            for &(u,_) in &self.graph[v] {
                indeg[u]+=1;
            }
        }
        let mut queue=std::collections::VecDeque::<usize>::new();
        for v in 0..self.size() {
            if indeg[v]==0 {
                queue.push_front(v);
            }
        }
        while let Some(v)=queue.pop_back() {
            ret.push(v);
            for &(u,_) in &self.graph[v] {
                indeg[u]-=1;
                if indeg[u]==0 {
                    queue.push_front(u);
                }
            }
        }
        if ret.len()==self.size() {
            Some(ret)
        } else {
            None
        }
    }
    /// グラフからUnion-Find木を構築する関数（0-indexed）
    pub fn construct_union_find(&self) -> ac_library::Dsu {
        let mut uf=ac_library::Dsu::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self.graph[v] {
                uf.merge(v, u);
            }
        }
        uf
    }
    /// クラスカル法を行い、全域木を返す関数（minimizeは最小全域木であるか（最大全域木ではないか）どうか）
    pub fn kruskal(&self, minimize: bool) -> Self {
        let mut ret=Self::new(self.size());
        let mut edge_cnt=0;
        let mut uf=ac_library::Dsu::new(self.size());
        let mut wvu=Vec::<(usize,usize,usize)>::new();
        for v in 0..self.size() {
            for &(u,w) in &self.graph[v] {
                if v<u {
                    wvu.push((w,v,u));
                }
            }
        }
        wvu.sort();
        if !minimize {
            wvu.reverse();
        }
        for i in 0..wvu.len() {
            let (w,v,u)=wvu[i];
            if !uf.same(v, u) {
                ret.get_mut()[v].push((u,w));
                ret.get_mut()[u].push((v,w));
                edge_cnt+=1;
                uf.merge(v, u);
            }
            if edge_cnt>=self.size()-1 {
                break;
            }
        }
        ret
    }
    /// プリム法を行い、全域木を返す関数（minimizeは最小全域木であるか（最大全域木ではないか）どうか）
    pub fn prim(&self, minimize: bool) -> Self {
        if minimize {
            let mut ret=Self::new(self.size());
            let mut edge_cnt=0;
            let mut pq=RevBinaryHeap::<(usize,usize,usize)>::new();
            let mut ok=vec![false;self.size()];
            ok[0]=true;
            for &(u,w) in &self.graph[0] {
                pq.push((w,0,u));
            }
            while edge_cnt<self.size()-1 {
                let (w,v,u)=pq.pop().unwrap();
                if ok[u] {
                    continue;
                }
                ret.get_mut()[v].push((u,w));
                ret.get_mut()[u].push((v,w));
                edge_cnt+=1;
                ok[u]=true;
                for &(t,w) in &self.graph[u] {
                    if !ok[t] {
                        pq.push((w,u,t));
                    }
                }
            }
            ret
        } else {
            let mut ret=Self::new(self.size());
            let mut edge_cnt=0;
            let mut pq=std::collections::BinaryHeap::<(usize,usize,usize)>::new();
            let mut ok=vec![false;self.size()];
            ok[0]=true;
            for &(u,w) in &self.graph[0] {
                pq.push((w,0,u));
            }
            while edge_cnt<self.size()-1 {
                let (w,v,u)=pq.pop().unwrap();
                if ok[u] {
                    continue;
                }
                ret.get_mut()[v].push((u,w));
                ret.get_mut()[u].push((v,w));
                edge_cnt+=1;
                ok[u]=true;
                for &(t,w) in &self.graph[u] {
                    if !ok[t] {
                        pq.push((w,u,t));
                    }
                }
            }
            ret
        }
    }
    /// グラフが二部グラフであるかを判定し、二部グラフであれば色分けの例を返す関数（返り値はOption）
    pub fn is_bipartite_graph(&self) -> Option<Vec<bool>> {
        let mut ts=ac_library::TwoSat::new(self.size());
        for v in 0..self.size() {
            for &(u,_) in &self.graph[v] {
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
    /// 木のプリューファーコードを返す関数（0-indexed）（グラフが無向木でない場合の動作は保証しない）
    pub fn pruefer_code(&self) -> Vec<usize> {
        let n=self.size();
        let mut adjacency_list=vec![std::collections::BTreeSet::<usize>::new();n];
        let mut d=vec![0;n];
        for v in 0..n {
            for &(u,_) in &self.graph[v] {
                adjacency_list[v].insert(u);
                d[v]+=1;
            }
        }
        let mut leaves=RevBinaryHeap::<usize>::new();
        for v in 0..n {
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let mut pc=vec![0;n-2];
        for i in 0..n-2 {
            let v=leaves.pop().unwrap();
            let u=adjacency_list[v].pop_first().unwrap();
            pc[i]=u;
            adjacency_list[u].remove(&v);
            d[v]-=1;
            d[u]-=1;
            if d[u]==1 {
                leaves.push(u);
            }
        }
        pc
    }
}

/// BTreeMapのベクターによるグラフの型（隣接の高速な判定が目的の型であるため、多重辺には対応していない）
#[derive(Clone, Default, Debug)]
pub struct MapGraph {
    graph: Vec<std::collections::BTreeMap<usize,usize>>
}

impl MapGraph {
    /// グラフを初期化する関数
    pub fn new(n: usize) -> Self {
        Self { graph: vec![std::collections::BTreeMap::<usize,usize>::new();n] }
    }
    /// 頂点数を返す関数
    pub fn size(&self) -> usize {
        self.graph.len()
    }
    /// 隣接リストの参照を返す関数
    pub fn get(&self) -> &Vec<std::collections::BTreeMap<usize,usize>> {
        &self.graph
    }
    /// 隣接リストの可変参照を返す関数
    pub fn get_mut(&mut self) -> &mut Vec<std::collections::BTreeMap<usize,usize>> {
        &mut self.graph
    }
    /// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g=MapGraph::new(n);
        for &(a,b) in ab {
            g.graph[a].insert(b, 1);
            g.graph[b].insert(a, 1);
        }
        g
    }
    /// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_directed_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        debug_assert_eq!(ab.len(), m);
        let mut g=MapGraph::new(n);
        for &(a,b) in ab {
            g.graph[a].insert(b, 1);
        }
        g
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=MapGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].insert(b, w);
            g.graph[b].insert(a, w);
        }
        g
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=MapGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].insert(b, w);
        }
        g
    }
    /// DFSのイテレータを返す関数
    pub fn dfs(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=Vec::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some((&u,&w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=stack.pop() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 全ての辺を通るDFSのイテレータを返す関数
    pub fn dfs_all_edges(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=Vec::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some((&u,&w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u);
                    }
                    return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                }
                if let Some(v)=stack.pop() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 帰りにも頂点を訪れるDFSのイテレータを返す関数
    pub fn dfs_postorder(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut stack=vec![start+self.size()];
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some((&u,&w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        stack.push(u+self.size());
                        stack.push(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=stack.pop() {
                    if v<self.size() {
                        vertex=v;
                        it=self.graph[v].iter();
                        Some(GraphSearch::Vertex(true, v))
                    } else {
                        Some(GraphSearch::Vertex(false, v-self.size()))
                    }
                } else {
                    None
                }
            }
        })
    }
    /// BFSのイテレータを返す関数
    pub fn bfs(&self, start: usize) -> impl Iterator<Item=GraphSearch> + '_ {
        let mut seen=vec![false;self.size()];
        seen[start]=true;
        let mut queue=std::collections::VecDeque::<usize>::new();
        let mut vertex=start;
        let mut it=self.graph[start].iter();
        let mut first=true;
        std::iter::from_fn(move || {
            if first {
                first=false;
                Some(GraphSearch::Vertex(true, start))
            } else {
                while let Some((&u,&w))=it.next() {
                    if !seen[u] {
                        seen[u]=true;
                        queue.push_front(u);
                        return Some(GraphSearch::VertexEdgeWeight(vertex, u, w));
                    }
                }
                if let Some(v)=queue.pop_back() {
                    vertex=v;
                    it=self.graph[v].iter();
                    Some(GraphSearch::Vertex(true, v))
                } else {
                    None
                }
            }
        })
    }
    /// 頂点aから頂点bへの辺があるかどうかを判定し、辺があれば重みを返す関数（返り値はOption）
    pub fn weight(&self, a: usize, b: usize) -> Option<usize> {
        for (&u,&w) in &self.graph[a] {
            if u==b {
                return Some(w);
            }
        }
        None
    }
    /// 最短経路の距離を返す関数（is_weightedがtrueでダイクストラ法、falseでBFS）（到達不能ならばusize::MAXが入る）
    pub fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize> {
        let mut dist=vec![usize::MAX;self.size()];
        dist[start]=0;
        if is_weighted {
            let mut pq=RevBinaryHeap::<(usize,usize)>::new();
            pq.push((dist[start],start));
            while let Some((d,v))=pq.pop() {
                if dist[v]<d {
                    continue;
                }
                for (&u,&w) in &self.graph[v] {
                    if dist[v].saturating_add(w)<dist[u] {
                        dist[u]=dist[v].saturating_add(w);
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
                for (&u,&w) in &self.graph[v] {
                    debug_assert_eq!(w, 1);
                    if !seen[u] {
                        dist[u]=dist[v].saturating_add(w);
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        dist
    }
    /// ワーシャル・フロイド法の関数（到達不能ならばusize::MAXが入る）
    pub fn floyd_warshall(&self) -> Vec<Vec<usize>> {
        let mut dist=vec![vec![usize::MAX;self.size()];self.size()];
        for v in 0..self.size() {
            for (&u,&w) in &self.graph[v] {
                dist[v][u]=w;
                dist[u][v]=w;
            }
        }
        for m in 0..self.size() {
            for v in 0..self.size() {
                for u in 0..self.size() {
                    dist[v][u]=min(dist[v][m].saturating_add(dist[m][u]), dist[v][u]);
                }
            }
        }
        dist
    }
    /// グラフに閉路がなければ、頂点をトポロジカルソートした結果を返す関数（返り値はOption）
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut ret=Vec::<usize>::new();
        let mut indeg=vec![0;self.size()];
        for v in 0..self.size() {
            for (&u,_) in &self.graph[v] {
                indeg[u]+=1;
            }
        }
        let mut queue=std::collections::VecDeque::<usize>::new();
        for v in 0..self.size() {
            if indeg[v]==0 {
                queue.push_front(v);
            }
        }
        while let Some(v)=queue.pop_back() {
            ret.push(v);
            for (&u,_) in &self.graph[v] {
                indeg[u]-=1;
                if indeg[u]==0 {
                    queue.push_front(u);
                }
            }
        }
        if ret.len()==self.size() {
            Some(ret)
        } else {
            None
        }
    }
    /// グラフからUnion-Find木を構築する関数（0-indexed）
    pub fn construct_union_find(&self) -> ac_library::Dsu {
        let mut uf=ac_library::Dsu::new(self.size());
        for v in 0..self.size() {
            for (&u,_) in &self.graph[v] {
                uf.merge(v, u);
            }
        }
        uf
    }
    /// クラスカル法を行い、全域木を返す関数（minimizeは最小全域木であるか（最大全域木ではないか）どうか）
    pub fn kruskal(&self, minimize: bool) -> Self {
        let mut ret=Self::new(self.size());
        let mut edge_cnt=0;
        let mut uf=ac_library::Dsu::new(self.size());
        let mut wvu=Vec::<(usize,usize,usize)>::new();
        for v in 0..self.size() {
            for (&u,&w) in &self.graph[v] {
                if v<u {
                    wvu.push((w,v,u));
                }
            }
        }
        wvu.sort();
        if !minimize {
            wvu.reverse();
        }
        for i in 0..wvu.len() {
            let (w,v,u)=wvu[i];
            if !uf.same(v, u) {
                ret.get_mut()[v].insert(u,w);
                ret.get_mut()[u].insert(v,w);
                edge_cnt+=1;
                uf.merge(v, u);
            }
            if edge_cnt>=self.size()-1 {
                break;
            }
        }
        ret
    }
    /// プリム法を行い、全域木を返す関数（minimizeは最小全域木であるか（最大全域木ではないか）どうか）
    pub fn prim(&self, minimize: bool) -> Self {
        if minimize {
            let mut ret=Self::new(self.size());
            let mut edge_cnt=0;
            let mut pq=RevBinaryHeap::<(usize,usize,usize)>::new();
            let mut ok=vec![false;self.size()];
            ok[0]=true;
            for (&u,&w) in &self.graph[0] {
                pq.push((w,0,u));
            }
            while edge_cnt<self.size()-1 {
                let (w,v,u)=pq.pop().unwrap();
                if ok[u] {
                    continue;
                }
                ret.get_mut()[v].insert(u,w);
                ret.get_mut()[u].insert(v,w);
                edge_cnt+=1;
                ok[u]=true;
                for (&t,&w) in &self.graph[u] {
                    if !ok[t] {
                        pq.push((w,u,t));
                    }
                }
            }
            ret
        } else {
            let mut ret=Self::new(self.size());
            let mut edge_cnt=0;
            let mut pq=std::collections::BinaryHeap::<(usize,usize,usize)>::new();
            let mut ok=vec![false;self.size()];
            ok[0]=true;
            for (&u,&w) in &self.graph[0] {
                pq.push((w,0,u));
            }
            while edge_cnt<self.size()-1 {
                let (w,v,u)=pq.pop().unwrap();
                if ok[u] {
                    continue;
                }
                ret.get_mut()[v].insert(u,w);
                ret.get_mut()[u].insert(v,w);
                edge_cnt+=1;
                ok[u]=true;
                for (&t,&w) in &self.graph[u] {
                    if !ok[t] {
                        pq.push((w,u,t));
                    }
                }
            }
            ret
        }
    }
    /// グラフが二部グラフであるかを判定し、二部グラフであれば色分けの例を返す関数（返り値はOption）
    pub fn is_bipartite_graph(&self) -> Option<Vec<bool>> {
        let mut ts=ac_library::TwoSat::new(self.size());
        for v in 0..self.size() {
            for (&u,_) in &self.graph[v] {
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
    /// 木のプリューファーコードを返す関数（0-indexed）（グラフが無向木でない場合の動作は保証しない）
    pub fn pruefer_code(&self) -> Vec<usize> {
        let n=self.size();
        let mut adjacency_list=vec![std::collections::BTreeSet::<usize>::new();n];
        let mut d=vec![0;n];
        for v in 0..n {
            for (&u,_) in &self.graph[v] {
                adjacency_list[v].insert(u);
                d[v]+=1;
            }
        }
        let mut leaves=RevBinaryHeap::<usize>::new();
        for v in 0..n {
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let mut pc=vec![0;n-2];
        for i in 0..n-2 {
            let v=leaves.pop().unwrap();
            let u=adjacency_list[v].pop_first().unwrap();
            pc[i]=u;
            adjacency_list[u].remove(&v);
            d[v]-=1;
            d[u]-=1;
            if d[u]==1 {
                leaves.push(u);
            }
        }
        pc
    }
}

/// 負の重みの辺をもつグラフの構造体
#[derive(Clone, Default, Debug)]
pub struct IsizeGraph {
    graph: Vec<Vec<(usize,isize)>>
}

impl IsizeGraph {
    /// グラフを初期化する関数
    pub fn new(n: usize) -> Self {
        Self { graph: vec![Vec::<(usize,isize)>::new();n] }
    }
    /// 頂点数を返す関数
    pub fn size(&self) -> usize {
        self.graph.len()
    }
    /// 隣接リストの参照を返す関数
    pub fn get(&self) -> &Vec<Vec<(usize,isize)>> {
        &self.graph
    }
    /// 隣接リストの可変参照を返す関数
    pub fn get_mut(&mut self) -> &mut Vec<Vec<(usize,isize)>> {
        &mut self.graph
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,isize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=IsizeGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, w));
            g.graph[b].push((a, w));
        }
        g
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    pub fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,isize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=IsizeGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, w));
        }
        g
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から、辺の重みの符号を反転した隣接リストを構築する関数（0-indexed）
    pub fn construct_inversed_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,isize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=IsizeGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, -w));
            g.graph[b].push((a, -w));
        }
        g
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から、辺の重みの符号を反転した隣接リストを構築する関数（0-indexed）
    pub fn construct_inversed_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,isize)>) -> Self {
        debug_assert_eq!(abw.len(), m);
        let mut g=IsizeGraph::new(n);
        for &(a,b,w) in abw {
            g.graph[a].push((b, -w));
        }
        g
    }
    /// ベルマン・フォード法を行い、始点から到達できる負の閉路がなければそれぞれの頂点との最短距離を返す関数（返り値はOptionで、到達不能ならばusize::MAXが入る）
    pub fn rough_bellman_ford(&self, start: usize) -> Option<Vec<isize>> {
        let mut dist=vec![isize::MAX;self.size()];
        let mut reached=vec![false;self.size()];
        dist[start]=0;
        reached[start]=true;
        for _ in 0..self.size()-1 {
            for v in 0..self.size() {
                for &(u,w) in &self.graph[v] {
                    if reached[v] {
                        if dist[v].saturating_add(w)<dist[u] {
                            dist[u]=dist[v].saturating_add(w);
                        }
                        reached[u]=true;
                    }
                }
            }
        }
        for v in 0..self.size() {
            for &(u,w) in &self.graph[v] {
                if reached[v] {
                    if dist[v].saturating_add(w)<dist[u] {
                        return None;
                    }
                }
            }
        }
        Some(dist)
    }
    /// ベルマン・フォード法を行い、それぞれの頂点との最短距離を返す関数（返り値はOptionのベクターで、到達不能ならばusize::MAXが入り、経路内に負の閉路があればNoneを返す）
    pub fn bellman_ford(&self, start: usize) -> Vec<Option<isize>> {
        let mut dist=vec![isize::MAX;self.size()];
        let mut reached=vec![false;self.size()];
        dist[start]=0;
        reached[start]=true;
        for _ in 0..self.size()-1 {
            for v in 0..self.size() {
                for &(u,w) in &self.graph[v] {
                    if reached[v] {
                        if dist[v].saturating_add(w)<dist[u] {
                            dist[u]=dist[v].saturating_add(w);
                        }
                        reached[u]=true;
                    }
                }
            }
        }
        let mut ret=vec![None;self.size()];
        for v in 0..self.size() {
            ret[v]=Some(dist[v]);
        }
        for _ in 0..self.size()-1 {
            for v in 0..self.size() {
                for &(u,w) in &self.graph[v] {
                    if reached[v] {
                        if ret[v].is_none() || dist[v].saturating_add(w)<dist[u] {
                            ret[u]=None;
                        }
                    }
                }
            }
        }
        ret
    }
}

/// 二分探索の関数（整数）
pub fn binary_search<T,F>(ok: T, bad: T, determine: F) -> T where T: num::PrimInt, F: Fn(T) -> bool {
    let right=ok>bad;
    let mut ok=ok;
    let mut bad=bad;
    while if right {
        ok-bad>T::one()
    } else {
        bad-ok>T::one()
    } {
        let mid=(ok+bad)/(T::one()+T::one());
        if determine(mid) {
            ok=mid;
        } else {
            bad=mid;
        }
    }
    ok
}

/// 二分探索の関数（f64）
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

/// 二分探索の関数（usize）（ok_is_smallはok<badであるかどうか、-1が答えの場合はmaxが返る）
pub fn usize_binary_search<F>(max: usize, ok_is_small: bool, determine: F) -> usize where F: Fn(usize) -> bool {
    if ok_is_small {
        let ret=binary_search(-1, max as isize, |mid| {
            determine(mid as usize)
        });
        if ret>=0 {
            ret as usize
        } else {
            max
        }
    } else {
        let ret=binary_search(max as isize, -1, |mid| {
            determine(mid as usize)
        }) as usize;
        ret
    }
}

/// 広義の尺取り法を行う関数（increaseは左側の値に対して右側の値が単調増加であるか、satisfiedは返す境界がdetermineを満たすかどうか）
/// （返り値はイテレータで、各lに対するrは-1が答えの場合mが返る）
pub fn two_pointers<F>(n: usize, m: usize, increase: bool, satisfied: bool, determine: &F) -> impl Iterator<Item=(usize,usize)> + '_ where F: Fn(usize,usize) -> bool {
    let mut r=if increase {
        -1
    } else {
        m as isize
    };
    (0..n).map(move |l| {
        if increase {
            while ((r+1) as usize)<m && determine(l,(r+1) as usize) {
                r+=1;
            }
            if satisfied {
                if r>=0 {
                    (l,r as usize)
                } else {
                    (l,m)
                }
            } else {
                (l,(r+1) as usize)
            }
        } else {
            while r-1>=0 && determine(l,(r-1) as usize) {
                r-=1;
            }
            if satisfied {
                (l,r as usize)
            } else {
                if r-1>=0 {
                    (l,(r-1) as usize)
                } else {
                    (l,m)
                }
            }
        }
    })
}

/// 最小値を取り出すことのできる優先度つきキューの構造体
#[derive(Clone, Default, Debug)]
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
    pub fn peek(&self) -> Option<&T> {
        if !self.is_empty() {
            let std::cmp::Reverse(ret)=self.binary_heap.peek().unwrap();
            Some(ret)
        } else {
            None
        }
    }
    pub fn clear(&mut self) {
        self.binary_heap.clear();
    }
}

/// 最大値を高速に取り出せるとともに、最大値以外の値も削除できる優先度つきキューの構造体
#[derive(Clone, Default, Debug)]
pub struct RemovableBinaryHeap<T> where T: Ord {
    binary_heap: std::collections::BinaryHeap<T>,
    removed: std::collections::BinaryHeap<T>
}

impl<T> RemovableBinaryHeap<T> where T: Ord {
    pub fn new() -> Self {
        Self {
            binary_heap: std::collections::BinaryHeap::<T>::new(),
            removed: std::collections::BinaryHeap::<T>::new()
        }
    }
    /// removedに含まれる要素のうち、現在削除できるものを削除する関数
    pub fn flush(&mut self) {
        while let Some(rem)=self.removed.peek() {
            if let Some(top)=self.binary_heap.peek() {
                if top==rem {
                    self.binary_heap.pop();
                    self.removed.pop();
                } else {
                    break;
                }
            } else {
                self.removed.clear();
            }
        }
    }
    pub fn is_empty(&mut self) -> bool {
        self.flush();
        self.binary_heap.is_empty()
    }
    /// pushしていない要素をremoveすると正しくない値となる可能性があるので注意
    pub fn len(&mut self) -> usize {
        self.binary_heap.len()-self.removed.len()
    }
    pub fn push(&mut self, item: T) {
        self.binary_heap.push(item);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.flush();
        self.binary_heap.pop()
    }
    pub fn peek(&mut self) -> Option<&T> {
        self.flush();
        self.binary_heap.peek()
    }
    /// itemを削除する関数
    pub fn remove(&mut self, item: T) {
        self.removed.push(item);
        self.flush();
    }
    pub fn clear(&mut self) {
        self.binary_heap.clear();
        self.removed.clear();
    }
}

/// 最小値を高速に取り出せるとともに、最小値以外の値も削除できる優先度つきキューの構造体
#[derive(Clone, Default, Debug)]
pub struct RemovableRevBinaryHeap<T> where T: Ord {
    binary_heap: RevBinaryHeap<T>,
    removed: RevBinaryHeap<T>
}

impl<T> RemovableRevBinaryHeap<T> where T: Ord {
    pub fn new() -> Self {
        Self {
            binary_heap: RevBinaryHeap::<T>::new(),
            removed: RevBinaryHeap::<T>::new()
        }
    }
    /// removedに含まれる要素のうち、現在削除できるものを削除する関数
    pub fn flush(&mut self) {
        while let Some(rem)=self.removed.peek() {
            if let Some(top)=self.binary_heap.peek() {
                if top==rem {
                    self.binary_heap.pop();
                    self.removed.pop();
                } else {
                    break;
                }
            } else {
                self.removed.clear();
            }
        }
    }
    pub fn is_empty(&mut self) -> bool {
        self.flush();
        self.binary_heap.is_empty()
    }
    /// pushしていない要素をremoveすると正しくない値となる可能性があるので注意
    pub fn len(&mut self) -> usize {
        self.binary_heap.len()-self.removed.len()
    }
    pub fn push(&mut self, item: T) {
        self.binary_heap.push(item);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.flush();
        self.binary_heap.pop()
    }
    pub fn peek(&mut self) -> Option<&T> {
        self.flush();
        self.binary_heap.peek()
    }
    /// itemを削除する関数
    pub fn remove(&mut self, item: T) {
        self.removed.push(item);
        self.flush();
    }
    pub fn clear(&mut self) {
        self.binary_heap.clear();
        self.removed.clear();
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

/// プリミティブな整数型についてZeroとOneトレイトを記述するマクロ
macro_rules! zero_one {
    ($($ty:ty),*) => {
        $(
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
        )*
    }
}

zero_one!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

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

/// BTreeMapを用いて個数を管理する多重集合についての型
pub type MultiSet<T>=std::collections::BTreeMap<T, usize>;

/// BTreeMapを用いて個数を管理する多重集合のトレイト
pub trait MapMultiSet {
    /// 多重集合で管理する要素の型
    type T;
    /// 多重集合に1つvalを追加する関数（返り値は追加した後のvalの個数のOption）
    fn insert_one(&mut self, val: Self::T) -> Option<usize>;
    /// 多重集合から1つvalを削除する関数（返り値は削除した後のvalの個数のOption）
    fn remove_one(&mut self, val: Self::T) -> Option<usize>;
}

impl<T> MapMultiSet for std::collections::BTreeMap<T, usize> where T: Copy + Ord {
    type T = T;
    fn insert_one(&mut self, val: Self::T) -> Option<usize> {
        if self.contains_key(&val) {
            self.insert(val, self[&val]+1);
        } else {
            self.insert(val, 1);
        }
        Some(self[&val])
    }
    fn remove_one(&mut self, val: Self::T) -> Option<usize> {
        if self.contains_key(&val) {
            if self[&val]>1 {
                self.insert(val, self[&val]-1);
                Some(self[&val])
            } else {
                self.remove(&val);
                Some(0)
            }
        } else {
            self.remove(&val);
            None
        }
    }
}

/// 追加と削除とmexの管理ができる非負整数の多重集合の構造体
#[derive(Clone, Debug)]
pub struct MexMultiSet {
    max: usize,
    multiset: MultiSet<usize>,
    complement: std::collections::BTreeSet<usize>
}

impl MexMultiSet {
    /// 初期化の関数（nは多重集合の重ならない最大の要素数）
    pub fn new(n: usize) -> Self {
        MexMultiSet {
            max: n,
            multiset: MultiSet::<usize>::new(),
            complement: std::iter::FromIterator::from_iter(0..=n)
        }
    }
    /// 多重集合に1つvalを追加する関数
    pub fn insert_one(&mut self, val: usize) {
        if val<=self.max {
            self.multiset.insert_one(val);
            self.complement.remove(&val);
        }
    }
    /// 多重集合から1つvalを削除する関数（返り値は削除する要素があったかどうか）
    pub fn remove_one(&mut self, val: usize) -> bool {
        if val<=self.max {
            let ret=self.multiset.remove_one(val);
            if let Some(cnt)=ret {
                if cnt==0 {
                    self.complement.insert(val);
                }
                true
            } else {
                false
            }
        } else {
            true
        }
    }
    /// 多重集合のmexを返す関数
    pub fn mex(&self) -> usize {
        *self.complement.first().unwrap()
    }
}

/// BTreeSetを用いて区間を管理する集合についての型
pub type RangeSet<T>=std::collections::BTreeSet<(T,T)>;

/// 区間を管理する集合のトレイト（現状は区間追加と区間削除には非対応）
pub trait SetOfRanges<T> {
    /// 初期化する関数
    fn initialize() -> Self;
    /// valが集合に属するかどうかを判定し、属するならば属する区間を返す関数（返り値はOption）
    fn includes(&self, val: T) -> Option<(T,T)>;
    /// valを集合に追加する関数（返り値は追加できたかどうか）
    fn insert_one(&mut self, val: T) -> bool;
    /// valを集合から削除する関数（返り値は削除できたかどうか）
    fn remove_one(&mut self, val: T) -> bool;
    /// 集合のmexを返す関数（isizeの場合も非負整数に限って考える）
    fn mex(&self) -> T;
}

impl SetOfRanges<usize> for std::collections::BTreeSet<(usize,usize)> {
    fn initialize() -> Self {
        std::collections::BTreeSet::default()
    }
    fn includes(&self, val: usize) -> Option<(usize,usize)> {
        if let Some(&range)=self.range(..(val+1,val+1)).last() {
            if val<=range.1 {
                Some(range)
            } else {
                None
            }
        } else {
            None
        }
    }
    fn insert_one(&mut self, val: usize) -> bool {
        if self.includes(val).is_none() {
            if val==0 {
                if let Some((_,r))=self.includes(1) {
                    self.remove(&(1,r));
                    self.insert((0,r));
                } else {
                    self.insert((0,0));
                }
            } else if let Some((l,_))=self.includes(val-1) {
                if let Some((_,r))=self.includes(val+1) {
                    self.remove(&(l,val-1));
                    self.remove(&(val+1,r));
                    self.insert((l,r));
                } else {
                    self.remove(&(l,val-1));
                    self.insert((l,val));
                }
            } else {
                if let Some((_,r))=self.includes(val+1) {
                    self.remove(&(val+1,r));
                    self.insert((val,r));
                } else {
                    self.insert((val,val));
                }
            }
            true
        } else {
            false
        }
    }
    fn remove_one(&mut self, val: usize) -> bool {
        if let Some((l,r))=self.includes(val) {
            self.remove(&(l,r));
            if l!=val {
                if r!=val {
                    self.insert((l,val-1));
                    self.insert((val+1,r));
                } else {
                    self.insert((l,val-1));
                }
            } else if r!=val {
                self.insert((val+1,r));
            }
            true
        } else {
            false
        }
    }
    fn mex(&self) -> usize {
        if let Some((_,r))=self.includes(0) {
            r+1
        } else {
            0
        }
    }
}

impl SetOfRanges<isize> for std::collections::BTreeSet<(isize,isize)> {
    fn initialize() -> Self {
        std::collections::BTreeSet::default()
    }
    fn includes(&self, val: isize) -> Option<(isize,isize)> {
        if let Some(&range)=self.range(..(val+1,val+1)).last() {
            if val<=range.1 {
                Some(range)
            } else {
                None
            }
        } else {
            None
        }
    }
    fn insert_one(&mut self, val: isize) -> bool {
        if self.includes(val).is_none() {
            if let Some((l,_))=self.includes(val-1) {
                if let Some((_,r))=self.includes(val+1) {
                    self.remove(&(l,val-1));
                    self.remove(&(val+1,r));
                    self.insert((l,r));
                } else {
                    self.remove(&(l,val-1));
                    self.insert((l,val));
                }
            } else {
                if let Some((_,r))=self.includes(val+1) {
                    self.remove(&(val+1,r));
                    self.insert((val,r));
                } else {
                    self.insert((val,val));
                }
            }
            true
        } else {
            false
        }
    }
    fn remove_one(&mut self, val: isize) -> bool {
        if let Some((l,r))=self.includes(val) {
            self.remove(&(l,r));
            if l!=val {
                if r!=val {
                    self.insert((l,val-1));
                    self.insert((val+1,r));
                } else {
                    self.insert((l,val-1));
                }
            } else if r!=val {
                self.insert((val+1,r));
            }
            true
        } else {
            false
        }
    }
    fn mex(&self) -> isize {
        if let Some((_,r))=self.includes(0) {
            r+1
        } else {
            0
        }
    }
}

/// ランレングス圧縮のトレイト
pub trait RunLengthEncoding {
    /// 配列やベクターをランレングス圧縮して、各要素とその連長の組のベクターを返す関数
    fn rle(&self) -> Vec<(Self::Output,usize)> where Self: std::ops::Index<usize>, Self::Output: Sized + Clone + PartialEq;
}

impl<T> RunLengthEncoding for Vec<T> {
    fn rle(&self) -> Vec<(<Vec<T> as std::ops::Index<usize>>::Output,usize)> where Self: std::ops::Index<usize>, <Vec<T> as std::ops::Index<usize>>::Output: Sized + Clone + PartialEq {
        let mut rle=Vec::<(<Vec<T> as std::ops::Index<usize>>::Output,usize)>::new();
        for i in 0..self.len() {
            if i>0 && self[i]==self[i-1] {
                rle.last_mut().unwrap().1+=1;
            } else {
                rle.push((self[i].clone(),1));
            }
        }
        rle
    }
}

impl<T> RunLengthEncoding for [T] {
    fn rle(&self) -> Vec<(<[T] as std::ops::Index<usize>>::Output,usize)> where Self: std::ops::Index<usize>, <[T] as std::ops::Index<usize>>::Output: Sized + Clone + PartialEq {
        let mut rle=Vec::<(<[T] as std::ops::Index<usize>>::Output,usize)>::new();
        for i in 0..self.len() {
            if i>0 && self[i]==self[i-1] {
                rle.last_mut().unwrap().1+=1;
            } else {
                rle.push((self[i].clone(),1));
            }
        }
        rle
    }
}

impl<T, const N: usize> RunLengthEncoding for [T;N] {
    fn rle(&self) -> Vec<(<[T;N] as std::ops::Index<usize>>::Output,usize)> where Self: std::ops::Index<usize>, <[T;N] as std::ops::Index<usize>>::Output: Sized + Clone + PartialEq {
        let mut rle=Vec::<(<[T;N] as std::ops::Index<usize>>::Output,usize)>::new();
        for i in 0..self.len() {
            if i>0 && self[i]==self[i-1] {
                rle.last_mut().unwrap().1+=1;
            } else {
                rle.push((self[i].clone(),1));
            }
        }
        rle
    }
}

/// 0以上max-1以下の整数の重複順列を全列挙する関数
pub fn permutations_with_replacement(max: usize, len: usize) -> Vec<Vec<usize>> {
    let mut ret=vec![vec![0;len];max.pow(len as u32)];
    for seq in 0..max.pow(len as u32) {
        let mut val=seq;
        for i in 0..len {
            ret[seq][i]=val%max;
            val/=max;
        }
    }
    ret
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
    fn legendre_s_formula(self, p: Self) -> Self;
    /// エラトステネスの篩で素数を列挙する関数
    fn sieve_of_eratosthenes(nmax: Self) -> Vec<bool>;
    /// 線形篩で最小素因数を列挙する関数
    fn linear_sieve(nmax: Self) -> (Vec<Self>,Vec<Self>);
    /// 線形篩を用いて素因数分解をする関数
    fn fast_prime_factorize(self, linear_sieve: &Vec<Self>) -> Vec<(Self,Self)>;
    /// 32bitで表せる非負整数についてミラー・ラビン素数判定法で素数か判定する関数
    fn is_prime_using_32bit_miller_rabin(self) -> bool;
    /// 64bitで表せる非負整数についてミラー・ラビン素数判定法で素数か判定する関数
    fn is_prime_using_64bit_miller_rabin(self) -> bool;
    /// ミラー・ラビン素数判定法で素数か判定する関数
    fn is_prime_using_miller_rabin(self) -> bool;
    /// 32bitで表せる非負整数についてポラード・ロー法で素因数分解をする関数
    fn prime_factorize_using_32bit_pollard_s_rho(self) -> Vec<(Self,Self)>;
    /// 64bitで表せる非負整数についてポラード・ロー法で素因数分解をする関数
    fn prime_factorize_using_64bit_pollard_s_rho(self) -> Vec<(Self,Self)>;
    /// ポラード・ロー法で素因数分解をする関数
    fn prime_factorize_using_pollard_s_rho(self) -> Vec<(Self,Self)>;
    /// ポラード・ロー法で素数冪か判定する関数
    fn is_prime_power_using_pollard_s_rho(self) -> bool;
    /// ポラード・ロー法で約数を列挙する関数
    fn enumerate_divisors_using_pollard_s_rho(self) -> Vec<Self>;
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
    fn legendre_s_formula(mut self, p: Self) -> Self {
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
    fn is_prime_using_32bit_miller_rabin(self) -> bool {
        debug_assert!(self<(1<<32));
        if self==2 || self==7 || self==61 {
            return true;
        }
        if self%2==0 || self==1 {
            return false;
        }
        let checkers=[2,7,61];
        let n=self;
        let m=n-1;
        let s=m.trailing_zeros();
        let d=m>>s;
        for a in checkers {
            let mut x=1;
            let mut p=a%n;
            let mut e=d;
            while e>0 {
                if e%2>0 {
                    x*=p;
                    x%=n;
                }
                p*=p;
                p%=n;
                e/=2;
            }
            if x==1 {
                continue;
            }
            let mut comp=true;
            for _ in 0..s {
                if x==m {
                    comp=false;
                    break;
                }
                x*=x;
                x%=n;
            }
            if comp {
                return false;
            }
        }
        true
    }
    fn is_prime_using_64bit_miller_rabin(self) -> bool {
        if self==2 {
            return true;
        }
        if self%2==0 || self==1 {
            return false;
        }
        let checkers=[2,325,9375,28178,450775,9780504,1795265022];
        let n=self as u128;
        let m=self-1;
        let s=m.trailing_zeros();
        let d=m>>s;
        for a in checkers {
            let mut x=1;
            let mut p=a as u128%n;
            let mut e=d;
            while e>0 {
                if e%2>0 {
                    x*=p;
                    x%=n;
                }
                p*=p;
                p%=n;
                e/=2;
            }
            if x==1 {
                continue;
            }
            let mut comp=true;
            for _ in 0..s {
                if x==n-1 {
                    comp=false;
                    break;
                }
                x*=x;
                x%=n;
            }
            if comp {
                return false;
            }
        }
        true
    }
    fn is_prime_using_miller_rabin(self) -> bool {
        if self<(1<<32) {
            self.is_prime_using_32bit_miller_rabin()
        } else {
            self.is_prime_using_64bit_miller_rabin()
        }
    }
    fn prime_factorize_using_32bit_pollard_s_rho(self) -> Vec<(Self,Self)> {
        debug_assert!(self<(1<<32));
        if self==1 {
            return Vec::new();
        }
        let cnt=self.trailing_zeros() as usize;
        if cnt>0 {
            let mut ret=vec![(2,cnt)];
            ret.extend((self>>cnt).prime_factorize_using_32bit_pollard_s_rho());
            return ret;
        }
        if self.is_prime_using_32bit_miller_rabin() {
            return vec![(self,1)];
        }
        let m=num_integer::Roots::nth_root(&self, 8)+1;
        let mut d=0;
        for c in 1..self {
            let f=|y:usize| ((y*y)%self+c)%self;
            let mut y=0;
            let mut r=1;
            let mut q=1;
            let mut g=1;
            let mut x=y;
            let mut s=y;
            while g==1 {
                x=y;
                for _ in 0..r {
                    y=f(y);
                }
                let mut k=0;
                while k<r && g==1 {
                    s=y;
                    for _ in 0..min(m,r-k) {
                        y=f(y);
                        q*=x.abs_diff(y);
                        q%=self;
                    }
                    g=num_integer::gcd(q, self);
                    k+=m;
                }
                r*=2;
            }
            if g==self {
                g=1;
                y=s;
                while g==1 {
                    y=f(y);
                    g=num_integer::gcd(x.abs_diff(y), self);
                }
            }
            if g<self {
                d=g;
                break;
            }
        }
        if d==0 {
            panic!("{}",self);
        }
        let mut ret1=d.prime_factorize_using_32bit_pollard_s_rho();
        let mut q=self/d;
        for pe in &mut ret1 {
            while q%pe.0==0 {
                q/=pe.0;
                pe.1+=1;
            }
        }
        merge_vecs(&ret1, &q.prime_factorize_using_32bit_pollard_s_rho())
    }
    fn prime_factorize_using_64bit_pollard_s_rho(self) -> Vec<(Self,Self)> {
        if self==1 {
            return Vec::new();
        }
        let cnt=self.trailing_zeros() as usize;
        if cnt>0 {
            let mut ret=vec![(2,cnt)];
            ret.extend((self>>cnt).prime_factorize_using_pollard_s_rho());
            return ret;
        }
        if self.is_prime_using_miller_rabin() {
            return vec![(self,1)];
        }
        let m=num_integer::Roots::nth_root(&self, 8)+1;
        let mut d=0;
        let n=self as u128;
        for c in 1..n {
            let f=|y:u128| ((y*y)%n+c)%n;
            let mut y=0;
            let mut r=1;
            let mut q=1;
            let mut g=1;
            let mut x=y;
            let mut s=y;
            while g==1 {
                x=y;
                for _ in 0..r {
                    y=f(y);
                }
                let mut k=0;
                while k<r && g==1 {
                    s=y;
                    for _ in 0..min(m,r-k) {
                        y=f(y);
                        q*=x.abs_diff(y);
                        q%=n;
                    }
                    g=num_integer::gcd(q, n);
                    k+=m;
                }
                r*=2;
            }
            if g==n {
                g=1;
                y=s;
                while g==1 {
                    y=f(y);
                    g=num_integer::gcd(x.abs_diff(y), n);
                }
            }
            if g<n {
                d=g as usize;
                break;
            }
        }
        if d==0 {
            panic!("{}",self);
        }
        let mut ret1=d.prime_factorize_using_pollard_s_rho();
        let mut q=self/d;
        for pe in &mut ret1 {
            while q%pe.0==0 {
                q/=pe.0;
                pe.1+=1;
            }
        }
        merge_vecs(&ret1, &q.prime_factorize_using_pollard_s_rho())
    }
    fn prime_factorize_using_pollard_s_rho(self) -> Vec<(Self,Self)> {
        if self<(1<<32) {
            self.prime_factorize_using_32bit_pollard_s_rho()
        } else {
            self.prime_factorize_using_64bit_pollard_s_rho()
        }
    }
    fn is_prime_power_using_pollard_s_rho(self) -> bool {
        if self<(1<<32) {
            self.prime_factorize_using_32bit_pollard_s_rho().len()==1
        } else {
            self.prime_factorize_using_64bit_pollard_s_rho().len()==1
        }
    }
    fn enumerate_divisors_using_pollard_s_rho(self) -> Vec<Self> {
        let pe=self.prime_factorize_using_pollard_s_rho();
        let mut ret=vec![1];
        for (p,e) in pe {
            let len=ret.len();
            for i in 0..len {
                let mut q=p;
                for _ in 0..e {
                    ret.push(ret[i]*q);
                    q*=p;
                }
            }
        }
        ret.sort();
        ret
    }
}

/// エラトステネスの篩で素数を列挙する関数
pub fn sieve_of_eratosthenes<T>(nmax: T) -> Vec<bool> where T: Primes {
    T::sieve_of_eratosthenes(nmax)
}

/// 線形篩で最小素因数を列挙する関数
pub fn linear_sieve<T>(nmax: T) -> (Vec<T>,Vec<T>) where T: Primes {
    T::linear_sieve(nmax)
}

/// 拡張ユークリッド互除法を行う関数（返り値はgcd(a,b)であり、x,yにはax+by=gcd(a,b)の1つの解が入る）
pub fn extended_euclidean_algorithm<T>(a: T, b: T, x: &mut isize, y: &mut isize) -> T where T: num::PrimInt {
    let mut d=a;
    if b!=T::zero() {
        d=extended_euclidean_algorithm(b, a%b, y, x);
        *y-=(a/b).to_isize().unwrap()*(*x);
    } else {
        *x=1;
        *y=0;
    }
    d
}

/// 平方剰余のトレイト
pub trait QuadraticResidue where Self: Sized {
    /// 平方剰余か判定する関数
    fn is_quadratic_residue(&self) -> bool;
    /// 2乗した剰余がselfになる数の片方を返す関数（返り値はOption）
    fn mod_sqrt(&self) -> Option<Self>;
}

impl<M> QuadraticResidue for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn is_quadratic_residue(&self) -> bool {
        debug_assert!(M::HINT_VALUE_IS_PRIME);
        self.pow((M::VALUE as u64-1)/2)==Self::new(1)
    }
    fn mod_sqrt(&self) -> Option<Self> {
        if *self==Self::new(0) || M::VALUE==2 {
            return Some(self.clone());
        }
        if !self.is_quadratic_residue() {
            return None;
        }
        if M::VALUE%4==3 {
            return Some(self.pow((M::VALUE as u64+1)/4));
        }
        let mut b=Self::new(1);
        while b.pow((M::VALUE as u64-1)/2)==Self::new(1) {
            b+=1;
        }
        let mut q=M::VALUE as usize-1;
        let q_digit=q.trailing_zeros();
        q>>=q_digit;
        let mut x=self.pow((q as u64+1)/2);
        b=b.pow(q as u64);
        let mut shift=2;
        while x*x!=*self {
            let err=self.inv()*x*x;
            if err.pow(1<<(q_digit-shift))!=Self::new(1) {
                x*=b;
            }
            b*=b;
            shift+=1;
        }
        if x.val()<(-x).val() {
            Some(x)
        } else {
            Some(-x)
        }
    }
}

impl<I> QuadraticResidue for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn is_quadratic_residue(&self) -> bool {
        self.pow((Self::modulus() as u64-1)/2)==Self::new(1)
    }
    fn mod_sqrt(&self) -> Option<Self> {
        if !self.is_quadratic_residue() {
            return None;
        }
        if *self==Self::new(0) || Self::modulus()==2 {
            return Some(self.clone());
        }
        if Self::modulus()%4==3 {
            return Some(self.pow((Self::modulus() as u64+1)/4));
        }
        let mut b=Self::new(1);
        while b.pow((Self::modulus() as u64-1)/2)==Self::new(1) {
            b+=1;
        }
        let mut q=Self::modulus() as usize-1;
        let q_digit=q.trailing_zeros();
        q>>=q_digit;
        let mut x=self.pow((q as u64+1)/2);
        b=b.pow(q as u64);
        let mut shift=2;
        while x*x!=*self {
            let err=self.inv()*x*x;
            if err.pow(1<<(q_digit-shift))!=Self::new(1) {
                x*=b;
            }
            b*=b;
            shift+=1;
        }
        if x.val()<(-x).val() {
            Some(x)
        } else {
            Some(-x)
        }
    }
}

/// 2つ以上の数の最大公約数を返すマクロ
#[macro_export]
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

/// modintなどで加算を行うモノイドの構造体
pub struct Add<S>(std::marker::PhantomData<S>);

/// modintなどで加算を行うモノイドのトレイト
impl<S> ac_library::Monoid for Add<S> where S: Copy + std::ops::Add<Output=S> + Zero {
    type S = S;
    fn identity() -> Self::S {
        S::zero_val()
    }
    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
        *a+*b
    }
}

/// modintなどで乗算を行うモノイドの構造体
pub struct Mul<S>(std::marker::PhantomData<S>);

/// modintなどで乗算を行うモノイドのトレイト
impl<S> ac_library::Monoid for Mul<S> where S: Copy + std::ops::Mul<Output=S> + One {
    type S = S;
    fn identity() -> Self::S {
        S::one_val()
    }
    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
        *a**b
    }
}

/// 遅延セグ木を双対セグ木として使うためのダミーの二項演算の構造体
pub struct DummyOperation<S>(std::marker::PhantomData<S>);

/// 遅延セグ木を双対セグ木として使うためのダミーの二項演算の構造体
impl<S> ac_library::Monoid for DummyOperation<S> where S: Default + Clone {
    type S = S;
    fn identity() -> Self::S {
        S::default()
    }
    fn binary_operation(a: &Self::S, _b: &Self::S) -> Self::S {
        a.clone()
    }
}

/// 区間のそれぞれの要素にアフィン変換を行う作用の構造体
pub struct SegmentAffineTransform<M>(std::marker::PhantomData<M>);

/// 区間のそれぞれの要素にアフィン変換を行う作用のトレイト
impl<M> ac_library::MapMonoid for SegmentAffineTransform<M> where M: ac_library::Monoid, M::S: std::ops::Add<Output=M::S> + std::ops::Mul<Output=M::S> + Zero + One {
    type M = M;
    type F = (M::S,M::S);
    fn identity_map() -> Self::F {
        (M::S::one_val(),<M as ac_library::Monoid>::S::zero_val())
    }
    fn mapping(f: &Self::F, x: &<Self::M as ac_library::Monoid>::S) -> <Self::M as ac_library::Monoid>::S {
        f.0.clone()*x.clone()+f.1.clone()
    }
    fn composition(f: &Self::F, g: &Self::F) -> Self::F {
        (f.0.clone()*g.0.clone(),f.0.clone()*g.1.clone()+f.1.clone())
    }
}

/// Nimber（NimのGrundy数）についてのトレイト
pub trait Nimber {
    /// Nimber（NimのGrundy数）を求める関数（kが0の場合は個数制限なし、k>0の場合はk個以下の個数制限あり）
    fn nimber(&self, k: usize) -> usize;
}

impl Nimber for Vec<usize> {
    fn nimber(&self, k: usize) -> usize {
        let mut nimber=0;
        for &a in self {
            nimber^=if k==0 {
                a
            } else {
                a%(k+1)
            };
        }
        nimber
    }
}

/// 累積XORについてのトレイト
pub trait PrefixXOR {
    /// 累積XORのベクターを構築する関数（kが0の場合は通常の累積XOR、k>0の場合はk個以下の個数制限ありのNimber）
    fn construct_prefix_xor(&self, k: usize) -> Self;
    /// 構築した累積XORのベクターから部分XORを計算する関数（0-indexedの左閉右開区間）
    fn calculate_partial_xor(&self, l: usize, r: usize) -> Self::Output where Self: std::ops::Index<usize>;
}

impl PrefixXOR for Vec<usize> {
    fn construct_prefix_xor(&self, k: usize) -> Self {
        let mut prefix_xor=vec![0;self.len()+1];
        for i in 0..self.len() {
            prefix_xor[i+1]=prefix_xor[i]^if k==0 {
                self[i]
            } else {
                self[i]%(k+1)
            };
        }
        prefix_xor
    }
    fn calculate_partial_xor(&self, l: usize, r: usize) -> usize {
        debug_assert!(l < self.len());
        debug_assert!(r <= self.len());
        self[r]^self[l]
    }
}

/// データ構造のベクターについてマージテクを行うトレイト（現状はVec、BTreeSet、BinaryHeapにのみ実装）
pub trait VecWeightedUnionHeuristic {
    /// データ構造のベクターについてマージテクを行う関数（現状はVec、BTreeSet、BinaryHeapにのみ実装）
    fn merge(&mut self, from: usize, to: usize);
}

impl<T> VecWeightedUnionHeuristic for Vec<Vec<T>> {
    fn merge(&mut self, from: usize, to: usize) {
        if self[from].len()>self[to].len() {
            self.swap(from, to);
        }
        let tmp=std::mem::take(&mut self[from]);
        self[to].extend(tmp);
    }
}

impl<T> VecWeightedUnionHeuristic for Vec<std::collections::BTreeSet<T>> where std::collections::BTreeSet<T>: Extend<T> {
    fn merge(&mut self, from: usize, to: usize) {
        if self[from].len()>self[to].len() {
            self.swap(from, to);
        }
        let tmp=std::mem::take(&mut self[from]);
        self[to].extend(tmp);
    }
}

impl<T> VecWeightedUnionHeuristic for Vec<std::collections::BinaryHeap<T>> where T: Ord {
    fn merge(&mut self, from: usize, to: usize) {
        if self[from].len()>self[to].len() {
            self.swap(from, to);
        }
        let tmp=std::mem::take(&mut self[from]);
        self[to].extend(tmp);
    }
}

/// データ構造の組についてマージテクを行うトレイト（現状はVec、BTreeSet、BinaryHeapにのみ実装）
pub trait WeightedUnionHeuristic {
    /// データ構造の組についてマージテクを行う関数（現状はVec、BTreeSet、BinaryHeapにのみ実装）
    fn merge(from: &mut Self, to: &mut Self);
}

impl<T> WeightedUnionHeuristic for Vec<T> {
    fn merge(from: &mut Self, to: &mut Self) {
        if from.len()>to.len() {
            std::mem::swap(from, to);
        }
        let tmp=std::mem::take(from);
        to.extend(tmp);
    }
}

impl<T> WeightedUnionHeuristic for std::collections::BTreeSet<T> where Self: Extend<T> {
    fn merge(from: &mut Self, to: &mut Self) {
        if from.len()>to.len() {
            std::mem::swap(from, to);
        }
        let tmp=std::mem::take(from);
        to.extend(tmp);
    }
}

impl<T> WeightedUnionHeuristic for std::collections::BinaryHeap<T> where T: Ord {
    fn merge(from: &mut Self, to: &mut Self) {
        if from.len()>to.len() {
            std::mem::swap(from, to);
        }
        let tmp=std::mem::take(from);
        to.extend(tmp);
    }
}

/// 半分全列挙を行うトレイト（それぞれの関数の引数はeが和の単位元、sumが和の関数、valが比較する値）
pub trait MeetInTheMiddle where Self: Sized + std::ops::Index<usize>, Self::Output: Sized {
    /// 半分全列挙によりvalと一致する部分和が存在するか判定する関数
    fn meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> bool where F1: Fn(Self::Output,Self::Output) -> Self::Output;
    /// 半分全列挙によりval以上の部分和が存在するか判定し、存在するならばその最小値を返す関数（返り値の型はOption）
    fn min_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Self::Output> where F1: Fn(Self::Output,Self::Output) -> Self::Output;
    /// 半分全列挙によりval以下の部分和が存在するか判定し、存在するならばその最大値を返す関数（返り値の型はOption）
    fn max_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Self::Output> where F1: Fn(Self::Output,Self::Output) -> Self::Output;
    /// 半分全列挙によりvalと一致する部分和が存在するか判定し、存在するならばその例に各要素が含まれるかを返す関数（返り値の型はOption）
    fn example_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Vec<bool>> where F1: Fn(Self::Output,Self::Output) -> Self::Output;
}

impl<T> MeetInTheMiddle for Vec<T> where T: Copy + Sized + PartialOrd {
    fn meet_in_the_middle<F1>(&self, e: T, sum: F1, val: T) -> bool where F1: Fn(T,T) -> T {
        let len=self.len();
        let mid=self.len()/2;
        let mut left_set=vec![e];
        for i in 0..mid {
            let left_1=std::mem::take(&mut left_set);
            let left_2=vec_range(0, 1<<i, |j| sum(left_1[j],self[i]));
            left_set=merge_vecs(&left_1, &left_2);
        }
        let mut right_set=vec![e];
        for i in 0..len-mid {
            let right_1=std::mem::take(&mut right_set);
            let right_2=vec_range(0, 1<<i, |j| sum(right_1[j],self[mid+i]));
            right_set=merge_vecs(&right_1, &right_2);
        }
        for (l,r) in two_pointers(left_set.len(), right_set.len(), false, true, &|i,j| {
            sum(left_set[i],right_set[j])>=val
        }) {
            if r<right_set.len() {
                if sum(left_set[l],right_set[r])==val {
                    return true;
                }
            }
        }
        false
    }
    fn min_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Self::Output> where F1: Fn(Self::Output,Self::Output) -> Self::Output {
        let len=self.len();
        let mid=self.len()/2;
        let mut left_set=vec![e];
        for i in 0..mid {
            let left_1=std::mem::take(&mut left_set);
            let left_2=vec_range(0, 1<<i, |j| sum(left_1[j],self[i]));
            left_set=merge_vecs(&left_1, &left_2);
        }
        let mut right_set=vec![e];
        for i in 0..len-mid {
            let right_1=std::mem::take(&mut right_set);
            let right_2=vec_range(0, 1<<i, |j| sum(right_1[j],self[mid+i]));
            right_set=merge_vecs(&right_1, &right_2);
        }
        let mut ans=None;
        for (l,r) in two_pointers(left_set.len(), right_set.len(), false, true, &|i,j| {
            sum(left_set[i],right_set[j])>=val
        }) {
            if r<right_set.len() {
                let tmp=sum(left_set[l],right_set[r]);
                if ans.is_some() {
                    if tmp<ans.unwrap() {
                        ans=Some(tmp);
                    }
                } else {
                    ans=Some(tmp);
                }
            }
        }
        ans
    }
    fn max_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Self::Output> where F1: Fn(Self::Output,Self::Output) -> Self::Output {
        let len=self.len();
        let mid=self.len()/2;
        let mut left_set=vec![e];
        for i in 0..mid {
            let left_1=std::mem::take(&mut left_set);
            let left_2=vec_range(0, 1<<i, |j| sum(left_1[j],self[i]));
            left_set=merge_vecs(&left_1, &left_2);
        }
        let mut right_set=vec![e];
        for i in 0..len-mid {
            let right_1=std::mem::take(&mut right_set);
            let right_2=vec_range(0, 1<<i, |j| sum(right_1[j],self[mid+i]));
            right_set=merge_vecs(&right_1, &right_2);
        }
        let mut ans=None;
        for (l,r) in two_pointers(left_set.len(), right_set.len(), false, false, &|i,j| {
            sum(left_set[i],right_set[j])>val
        }) {
            if r<right_set.len() {
                let tmp=sum(left_set[l],right_set[r]);
                if ans.is_some() {
                    if tmp>ans.unwrap() {
                        ans=Some(tmp);
                    }
                } else {
                    ans=Some(tmp);
                }
            }
        }
        ans
    }
    fn example_using_meet_in_the_middle<F1>(&self, e: Self::Output, sum: F1, val: Self::Output) -> Option<Vec<bool>> where F1: Fn(Self::Output,Self::Output) -> Self::Output {
        let len=self.len();
        let mid=self.len()/2;
        let mut left_set=vec![(e,0usize)];
        for i in 0..mid {
            let left_1=std::mem::take(&mut left_set);
            let left_2=vec_range(0, 1<<i, |j| (sum(left_1[j].0,self[i]),left_1[j].1+(1<<i)));
            left_set=merge_vecs(&left_1, &left_2);
        }
        let mut right_set=vec![(e,0usize)];
        for i in 0..len-mid {
            let right_1=std::mem::take(&mut right_set);
            let right_2=vec_range(0, 1<<i, |j| (sum(right_1[j].0,self[mid+i]),right_1[j].1+(1<<(mid+i))));
            right_set=merge_vecs(&right_1, &right_2);
        }
        for (l,r) in two_pointers(left_set.len(), right_set.len(), false, true, &|i,j| {
            sum(left_set[i].0,right_set[j].0)>=val
        }) {
            if r<right_set.len() {
                if sum(left_set[l].0,right_set[r].0)==val {
                    let mut sets=left_set[l].1+right_set[r].1;
                    let mut ret=vec![false;len];
                    for i in 0..len {
                        ret[i]=sets%2>0;
                        sets/=2;
                    }
                    return Some(ret);
                }
            }
        }
        None
    }
}

/// N1×N2行列の構造体（num::powで行列累乗を計算できる）
#[derive(Clone, Debug)]
pub struct Matrix<T, const N1: usize, const N2: usize> {
    pub matrix: [[T;N2];N1]
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

/// 簡易的なハッシュの構造体
#[derive(Clone, Debug)]
pub struct Hashes<T> {
    rng: rand::rngs::ThreadRng,
    hash_numbers: std::collections::BTreeMap<T,usize>
}

impl<T> Hashes<T> where T: Clone + Ord {
    /// 初期化の関数
    pub fn new() -> Self {
        Hashes { rng: rand::thread_rng(), hash_numbers: std::collections::BTreeMap::<_,_>::default() }
    }
    /// 既にハッシュが決まっているかどうかを返す関数
    pub fn is_assigned_hash(&self, a: T) -> bool {
        self.hash_numbers.contains_key(&a)
    }
    /// aのハッシュを返す関数
    pub fn hash(&mut self, a: T) -> usize {
        if !self.hash_numbers.contains_key(&a) {
            self.hash_numbers.insert(a.clone(), rand::Rng::gen_range(&mut self.rng, 1..(1<<60)));
        }
        *self.hash_numbers.get(&a).unwrap()
    }
}

/// ローリングハッシュの剰余の定数
const ROLLING_HASH_MOD:usize=2_305_843_009_213_693_951;

/// ローリングハッシュの基数の構造体
#[derive(Clone, Debug)]
pub struct RollingHashBases<const N: usize> {
    pows: [Vec<usize>;N],
    invs: [Vec<usize>;N]
}

impl<const N: usize> RollingHashBases<N> {
    /// ローリングハッシュの基数をランダムに生成する関数
    pub fn gen(nmax: usize, sigma: usize) -> Self {
        const PRIMITIVE_ROOT:usize=37;
        let pow=|k:usize| {
            let mut b=1;
            let mut r=PRIMITIVE_ROOT as u128;
            let mut k=k;
            while k>0 {
                if k%2>0 {
                    b*=r;
                    b%=ROLLING_HASH_MOD as u128;
                }
                r*=r;
                r%=ROLLING_HASH_MOD as u128;
                k/=2;
            }
            b as usize
        };
        let mut rng=rand::thread_rng();
        let mut pows=[();N].map(|_| vec![1;nmax+2]);
        let mut invs=[();N].map(|_| vec![1;nmax+2]);
        for i in 0..N {
            let mut k=rand::Rng::gen_range(&mut rng, 0..ROLLING_HASH_MOD-1);
            let mut b=pow(k);
            while num_integer::gcd(k, ROLLING_HASH_MOD-1)>1 || b<=sigma {
                k=rand::Rng::gen_range(&mut rng, 0..ROLLING_HASH_MOD-1);
                b=pow(k);
            }
            let mut inv=0;
            let mut tmp=0;
            extended_euclidean_algorithm(b, ROLLING_HASH_MOD, &mut inv, &mut tmp);
            if inv<0 {
                inv+=-inv/ROLLING_HASH_MOD as isize*ROLLING_HASH_MOD as isize;
                inv+=ROLLING_HASH_MOD as isize;
            }
            let inv=inv as usize%ROLLING_HASH_MOD;
            pows[i][1]=b;
            invs[i][1]=inv;
            for j in 2..=nmax+1 {
                let mut next_pow=pows[i][j-1] as u128;
                next_pow*=b as u128;
                next_pow%=ROLLING_HASH_MOD as u128;
                pows[i][j]=next_pow as usize;
                let mut next_inv=invs[i][j-1] as u128;
                next_inv*=inv as u128;
                next_inv%=ROLLING_HASH_MOD as u128;
                invs[i][j]=next_inv as usize;
            }
        }
        Self { pows, invs }
    }
    /// 基数を返す関数
    pub fn bases(&self) -> [usize;N] {
        let mut ret=[0;N];
        for i in 0..N {
            ret[i]=self.pows[i][1];
        }
        ret
    }
    /// powsを参照できる関数
    pub fn get_pows(&self) -> &[Vec<usize>;N] {
        &self.pows
    }
    /// invsを参照できる関数
    pub fn get_invs(&self) -> &[Vec<usize>;N] {
        &self.invs
    }
}

/// 累積和によるローリングハッシュの構造体
#[derive(Clone, Debug)]
pub struct RollingHash<const N: usize> {
    hashes: [Vec<usize>;N]
}

impl<const N: usize> RollingHash<N> {
    /// ハッシュの累積和と基数およびその逆元から部分列のローリングハッシュを返す関数（0-indexedの左閉右開区間）
    pub fn rolling_hash_of_subsequence(&self, l: usize, r: usize, b: &RollingHashBases<N>) -> [usize;N] {
        let mut hash=[0;N];
        for i in 0..N {
            let n=self.hashes[i].len()-1;
            let mut tmp=self.hashes[i][r] as u128;
            tmp+=ROLLING_HASH_MOD as u128;
            tmp-=self.hashes[i][l] as u128;
            tmp%=ROLLING_HASH_MOD as u128;
            tmp*=b.invs[i][n-r] as u128;
            tmp%=ROLLING_HASH_MOD as u128;
            hash[i]=tmp as usize;
        }
        hash
    }
    /// ハッシュの累積和と基数およびその逆元から部分列を結合した列のローリングハッシュを返す関数（0-indexedの左閉右開区間）
    pub fn sum_of_rolling_hash_of_subsequences(&self, ranges: &Vec<(usize,usize)>, b: &RollingHashBases<N>) -> [usize;N] {
        let mut hash=[0;N];
        let mut tmp=[0;N];
        for &(l,r) in ranges {
            let h=self.rolling_hash_of_subsequence(l, r, b);
            for i in 0..N {
                tmp[i]*=b.pows[i][r-l] as u128;
                tmp[i]%=ROLLING_HASH_MOD as u128;
                tmp[i]+=h[i] as u128;
                tmp[i]%=ROLLING_HASH_MOD as u128;
            }
        }
        for i in 0..N {
            hash[i]=tmp[i] as usize;
        }
        hash
    }
}

/// 文字列や数列のトレイト
pub trait Sequence where Self: Sized {
    /// 列の中身の型
    type T;
    /// 累積和によるローリングハッシュを返す関数
    fn calculate_rolling_hashes<const N: usize>(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N>;
}

impl Sequence for Vec<char> {
    type T = char;
    fn calculate_rolling_hashes<const N: usize>(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
        let n=self.len();
        let mut hashes=[();N].map(|_| vec![0;n+1]);
        for i in 0..N {
            let mut hash=0;
            for j in 0..n {
                let c=(self[j] as u128-begin as u128+1)%ROLLING_HASH_MOD as u128;
                hash+=c*b.pows[i][n-1-j] as u128;
                hash%=ROLLING_HASH_MOD as u128;
                hashes[i][j+1]=hash as usize;
            }
        }
        RollingHash { hashes }
    }
}

impl Sequence for Vec<usize> {
    type T = usize;
    fn calculate_rolling_hashes<const N: usize>(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
        let n=self.len();
        let mut hashes=[();N].map(|_| vec![0;n+1]);
        for i in 0..N {
            let mut hash=0;
            for j in 0..n {
                let c=(self[j] as u128-begin as u128+1)%ROLLING_HASH_MOD as u128;
                hash+=c*b.pows[i][n-1-j] as u128;
                hash%=ROLLING_HASH_MOD as u128;
                hashes[i][j+1]=hash as usize;
            }
        }
        RollingHash { hashes }
    }
}

impl Sequence for String {
    type T = char;
    fn calculate_rolling_hashes<const N: usize>(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
        self.chars().collect::<Vec<_>>().calculate_rolling_hashes(begin, b)
    }
}

/// ローリングハッシュのモノイドの構造体
pub struct DynamicRollingHash<const N: usize>;

impl<const N: usize> ac_library::Monoid for DynamicRollingHash<N> {
    type S = ([usize;N],[usize;N]);
    fn identity() -> Self::S {
        ([0;N],[1;N])
    }
    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S {
        let &(a_hashes,a_pows)=a;
        let &(b_hashes,b_pows)=b;
        let mut hashes=[0;N];
        let mut pows=[1;N];
        for i in 0..N {
            let mut hash=a_hashes[i] as u128;
            hash*=b_pows[i] as u128;
            hash%=ROLLING_HASH_MOD as u128;
            hash+=b_hashes[i] as u128;
            hash%=ROLLING_HASH_MOD as u128;
            hashes[i]=hash as usize;
            let mut pow=a_pows[i] as u128;
            pow*=b_pows[i] as u128;
            pow%=ROLLING_HASH_MOD as u128;
            pows[i]=pow as usize;
        }
        (hashes,pows)
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

/// 重みつきUnion-Findの構造体（重みの型はisize）（0-indexed）
#[derive(Clone, Debug)]
pub struct WeightedDSU {
    parents: Vec<isize>,
    potentials: Vec<isize>
}

impl WeightedDSU {
    /// n頂点の重みつきUnion-Findを初期化する関数
    pub fn new(n: usize) -> Self {
        WeightedDSU { parents: vec![-1;n], potentials: vec![0;n] }
    }
    /// 頂点aの属する連結成分の代表元を返す関数
    pub fn leader(&mut self, a: usize) -> usize {
        if self.parents[a]<0 {
            return a;
        }
        let leader=self.leader(self.parents[a] as usize);
        self.potentials[a]+=self.potentials[self.parents[a] as usize];
        self.parents[a]=leader as isize;
        return leader;
    }
    /// 頂点a,bが連結かどうかを返す関数
    pub fn same(&mut self, a: usize, b: usize) -> bool {
        self.leader(a)==self.leader(b)
    }
    /// 頂点aのポテンシャルを返す関数
    pub fn potential(&mut self, a: usize) -> isize {
        self.leader(a);
        self.potentials[a]
    }
    /// 頂点aのポテンシャルから頂点bのポテンシャルを引いた値を返す関数
    pub fn dist(&mut self, a: usize, b: usize) -> isize {
        self.potential(a)-self.potential(b)
    }
    /// 現在のグラフと無矛盾である場合、頂点aと頂点bを連結にしてself.dist(a,b)の値をdistにする関数（返り値は現在のグラフと無矛盾であるかどうか）
    pub fn merge(&mut self, mut a: usize, mut b: usize, mut dist: isize) -> bool {
        dist+=self.potential(b)-self.potential(a);
        a=self.leader(a);
        b=self.leader(b);
        if a==b {
            return dist==0;
        }
        if self.parents[a]<self.parents[b] {
            std::mem::swap(&mut a, &mut b);
            dist*=-1;
        }
        self.parents[b]+=self.parents[a];
        self.parents[a]=b as isize;
        self.potentials[a]=dist;
        true
    }
    /// 頂点aの属する連結成分のサイズを返す関数
    pub fn size(&mut self, a: usize) -> usize {
        let leader=self.leader(a);
        (-self.parents[leader]) as usize
    }
}

/// slope trickの構造体
#[derive(Clone, Default, Debug)]
pub struct SlopeTrick<T> where T: Ord {
    min: T,
    left_vertices: std::collections::BinaryHeap<T>,
    right_vertices: RevBinaryHeap<T>,
    left_offset: T,
    right_offset: T
}

impl<T> SlopeTrick<T> where T: Default + num::PrimInt + std::ops::AddAssign {
    /// 初期化の関数
    pub fn new() -> Self {
        SlopeTrick::default()
    }
    /// 関数の最小値を返す関数
    pub fn min(&self) -> T {
        self.min
    }
    /// 最小値をとる点の最小値を返す関数（存在しなければNoneを返す）
    pub fn left_min_point(&self) -> Option<T> {
        if let Some(&p)=self.left_vertices.peek() {
            Some(p+self.left_offset)
        } else {
            None
        }
    }
    /// 最小値をとる点の最大値を返す関数（存在しなければNoneを返す）
    pub fn right_min_point(&self) -> Option<T> {
        if let Some(&p)=self.right_vertices.peek() {
            Some(p+self.right_offset)
        } else {
            None
        }
    }
    /// 関数に定数関数を足す関数
    pub fn add_const(&mut self, c: T) {
        self.min+=c;
    }
    /// 関数にmax{p-x,0}を足す関数
    pub fn add_left_slope(&mut self, p: T) {
        if let Some(mp)=self.right_min_point() {
            if p>mp {
                self.min+=p-mp;
                self.right_vertices.push(p-self.right_offset);
                self.right_vertices.pop();
                self.left_vertices.push(mp-self.left_offset);
            } else {
                self.left_vertices.push(p-self.left_offset);
            }
        } else {
            self.left_vertices.push(p-self.left_offset);
        }
    }
    /// 関数にmax{x-p,0}を足す関数
    pub fn add_right_slope(&mut self, p: T) {
        if let Some(mp)=self.left_min_point() {
            if p<mp {
                self.min+=mp-p;
                self.left_vertices.push(p-self.left_offset);
                self.left_vertices.pop();
                self.right_vertices.push(mp-self.right_offset);
            } else {
                self.right_vertices.push(p-self.right_offset);
            }
        } else {
            self.right_vertices.push(p-self.right_offset);
        }
    }
    /// 関数に|x-p|を足す関数
    pub fn add_abs_slope(&mut self, p: T) {
        self.add_left_slope(p);
        self.add_right_slope(p);
    }
    /// 関数の左側累積minをとる関数（破壊的であることに注意）
    pub fn prefix_min(&mut self) {
        self.right_vertices.clear();
    }
    /// 関数の右側累積minをとる関数（破壊的であることに注意）
    pub fn suffix_min(&mut self) {
        self.left_vertices.clear();
    }
    /// 関数f(x)をf(x-a)に変化させる関数
    pub fn shift(&mut self, a: T) {
        self.left_offset+=a;
        self.right_offset+=a;
    }
    /// 関数f(x)を区間[x-b,x-a]におけるfの最小値に変化させる関数
    pub fn sliding_window_minimum(&mut self, a: T, b: T) {
        debug_assert!(a<=b);
        self.left_offset+=a;
        self.right_offset+=b;
    }
}

/// NTT素数のベクターで形式的冪級数を扱うトレイト（計算によって次数が変わるものはdegで結果の次数を指定）
pub trait FPS where Self: Sized {
    /// 対応するSparseFPSの型
    type Sparse;
    /// 形式的冪級数の最初のlen項を割り当てる関数
    fn fps_prefix_assign(&mut self, len: usize);
    /// 形式的冪級数の最初のlen項を返す関数
    fn fps_prefix(&self, len: usize) -> Self;
    /// 形式的冪級数の和を割り当てる関数
    fn fps_add_assign(&mut self, g: &Self);
    /// 形式的冪級数の和を返す関数
    fn fps_add(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の差を割り当てる関数
    fn fps_sub_assign(&mut self, g: &Self);
    /// 形式的冪級数の差を返す関数
    fn fps_sub(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の定数倍を割り当てる関数
    fn fps_scalar_assign(&mut self, k: isize);
    /// 形式的冪級数の定数倍を返す関数
    fn fps_scalar(&self, k: isize) -> Self;
    /// 形式的冪級数の積を割り当てる関数
    fn fps_mul_assign(&mut self, g: &Self, deg: usize);
    /// 形式的冪級数の積を返す関数
    fn fps_mul(f: &Self, g: &Self, deg: usize) -> Self;
    /// 形式的冪級数の逆元を返す関数
    fn fps_inv(&self, deg: usize) -> Self;
    /// 形式的冪級数の商を割り当てる関数
    fn fps_div_assign(&mut self, g: &Self, deg: usize);
    /// 形式的冪級数の商を返す関数
    fn fps_div(f: &Self, g: &Self, deg: usize) -> Self;
    /// 形式的冪級数の導関数を割り当てる関数
    fn fps_diff_assign(&mut self);
    /// 形式的冪級数の導関数を返す関数
    fn fps_diff(&self) -> Self;
    /// 形式的冪級数の原始関数を割り当てる関数
    fn fps_int_assign(&mut self);
    /// 形式的冪級数の原始関数を返す関数
    fn fps_int(&self) -> Self;
    /// 形式的冪級数の対数を割り当てる関数
    fn fps_log_assign(&mut self, deg: usize);
    /// 形式的冪級数の対数を返す関数
    fn fps_log(&self, deg: usize) -> Self;
    /// 形式的冪級数の指数を割り当てる関数
    fn fps_exp_assign(&mut self, deg: usize);
    /// 形式的冪級数の指数を返す関数
    fn fps_exp(&self, deg: usize) -> Self;
    /// 形式的冪級数の冪を割り当てる関数
    fn fps_pow_assign(&mut self, k: usize, deg: usize);
    /// 形式的冪級数の冪を返す関数
    fn fps_pow(&self, k: usize, deg: usize) -> Self;
    /// 形式的冪級数の平方根を割り当てる関数
    fn fps_sqrt_assign(&mut self, deg: usize);
    /// 形式的冪級数の平方根を返す関数
    fn fps_sqrt(&self, deg: usize) -> Self;
    /// 疎な形式的冪級数との積を割り当てる関数
    fn sparse_fps_mul_assign(&mut self, g: &Self::Sparse, deg: usize);
    /// 疎な形式的冪級数との積を返す関数
    fn sparse_fps_mul(f: &Self, g: &Self::Sparse, deg: usize) -> Self;
    /// 疎な形式的冪級数による商を割り当てる関数
    fn sparse_fps_div_assign(&mut self, g: &Self::Sparse, deg: usize);
    /// 疎な形式的冪級数による商を返す関数
    fn sparse_fps_div(f: &Self, g: &Self::Sparse, deg: usize) -> Self;
    /// 形式的冪級数の配列を受け取ってその総積を計算し、配列を破壊して最初の要素に総積を代入する関数
    fn fps_prod_merge(fs: &mut Vec<Self>);
}

impl<M> FPS for Vec<ac_library::StaticModInt<M>> where M: ac_library::Modulus {
    type Sparse = Vec<(usize,ac_library::StaticModInt<M>)>;
    fn fps_prefix_assign(&mut self, len: usize) {
        self.resize(len, ac_library::StaticModInt::<M>::new(0));
    }
    fn fps_prefix(&self, len: usize) -> Self {
        if self.len()>=len {
            self[0..len].to_vec()
        } else {
            let mut h=self.clone();
            h.fps_prefix_assign(len);
            h
        }
    }
    fn fps_add_assign(&mut self, g: &Self) {
        self.fps_prefix_assign(max(self.len(),g.len()));
        for i in 0..g.len() {
            self[i]+=g[i];
        }
    }
    fn fps_add(f: &Self, g: &Self) -> Self {
        let mut h=f.clone();
        h.fps_add_assign(&g);
        h
    }
    fn fps_sub_assign(&mut self, g: &Self) {
        self.fps_prefix_assign(max(self.len(),g.len()));
        for i in 0..g.len() {
            self[i]-=g[i];
        }
    }
    fn fps_sub(f: &Self, g: &Self) -> Self {
        let mut h=f.clone();
        h.fps_sub_assign(&g);
        h
    }
    fn fps_scalar_assign(&mut self, k: isize) {
        for i in 0..self.len() {
            self[i]*=k;
        }
    }
    fn fps_scalar(&self, k: isize) -> Self {
        let mut h=self.clone();
        h.fps_scalar_assign(k);
        h
    }
    fn fps_mul_assign(&mut self, g: &Self, deg: usize) {
        debug_assert!(self.len()+g.len()-1>deg);
        *self=FPS::fps_mul(self, g, deg);
    }
    fn fps_mul(f: &Self, g: &Self, deg: usize) -> Self {
        debug_assert!(f.len()+g.len()-1>deg);
        ac_library::convolution::convolution(&f[0..], &g[0..])[0..=deg].to_vec()
    }
    fn fps_inv(&self, deg: usize) -> Self {
        debug_assert!(self.len()>deg);
        let mut inv=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        inv[0]=self[0].inv();
        let mut curlen=1;
        while curlen<=deg {
            curlen*=2;
            let h=&ac_library::convolution::convolution(&self[0..min(curlen,deg+1)], &inv[0..curlen/2])[0..curlen];
            let h=&ac_library::convolution::convolution(&h[curlen/2..curlen], &inv[0..curlen/2])[0..min(curlen,deg+1)-curlen/2];
            for i in curlen/2..min(curlen,deg+1) {
                inv[i]=-h[i-curlen/2];
            }
        }
        inv
    }
    fn fps_div_assign(&mut self, g: &Self, deg: usize) {
        debug_assert!(self.len()+g.len()-1>deg);
        self.fps_mul_assign(&g.fps_inv(g.len()-1), deg);
    }
    fn fps_div(f: &Self, g: &Self, deg: usize) -> Self {
        debug_assert!(f.len()+g.len()-1>deg);
        let mut h=f.clone();
        h.fps_div_assign(&g, deg);
        h
    }
    fn fps_diff_assign(&mut self) {
        let n=self.len()-1;
        for i in 0..n {
            self[i]=self[i+1]*(i+1);
        }
        self[n]=ac_library::StaticModInt::<M>::new(0);
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
    fn fps_log_assign(&mut self, deg: usize) {
        debug_assert!(self.len()>deg);
        *self=FPS::fps_log(self, deg);
    }
    fn fps_log(&self, deg: usize) -> Self {
        debug_assert!(self.len()>deg);
        debug_assert_eq!(self[0], ac_library::StaticModInt::<M>::new(1));
        let mut h=self.clone();
        h.fps_diff_assign();
        h.fps_div_assign(self, deg);
        h.fps_int_assign();
        h
    }
    fn fps_exp_assign(&mut self, deg: usize) {
        debug_assert!(self.len()>deg);
        *self=FPS::fps_exp(self, deg);
    }
    fn fps_exp(&self, deg: usize) -> Self {
        debug_assert!(self.len()>deg);
        debug_assert_eq!(self[0], ac_library::StaticModInt::<M>::new(0));
        let mut exp=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        exp[0]=ac_library::StaticModInt::<M>::new(1);
        let mut curlen=1;
        while curlen<=deg {
            curlen*=2;
            let h=exp.fps_prefix(min(curlen,deg+1));
            let mut l=h.fps_diff();
            l.fps_div_assign(&h, curlen-1);
            for i in (curlen/2..min(curlen,deg+1)).rev() {
                l[i]=l[i-1]/i;
                l[i]-=self[i];
            }
            let h=&ac_library::convolution::convolution(&l[curlen/2..curlen], &exp[0..curlen/2])[0..min(curlen,deg+1)-curlen/2];
            for i in curlen/2..min(curlen,deg+1) {
                exp[i]-=h[i-curlen/2];
            }
        }
        exp
    }
    fn fps_pow_assign(&mut self, k: usize, deg: usize) {
        debug_assert!((self.len()-1)*k>=deg);
        let n=self.len()-1;
        self.fps_prefix_assign(deg+1);
        let mut lower=(0,ac_library::StaticModInt::<M>::new(1));
        for i in 0..=n {
            if self[i]!=ac_library::StaticModInt::<M>::new(0) {
                lower=(i,self[i]);
                break;
            }
        }
        for i in 0..=n-lower.0 {
            self[i]=self[i+lower.0]/lower.1;
        }
        for i in n-lower.0+1..=n {
            self[i]=ac_library::StaticModInt::<M>::new(0);
        }
        self.fps_log_assign(deg);
        self.fps_scalar_assign(k as isize);
        self.fps_exp_assign(deg);
        let lpow=lower.1.pow(k as u64);
        for i in (lower.0*k..=deg).rev() {
            self[i]=self[i-lower.0*k]*lpow;
        }
        for i in 0..min(lower.0*k,deg+1) {
            self[i]=ac_library::StaticModInt::<M>::new(0);
        }
    }
    fn fps_pow(&self, k: usize, deg: usize) -> Self {
        debug_assert!((self.len()-1)*k>=deg);
        let mut h=self.clone();
        h.fps_pow_assign(k, deg);
        h
    }
    fn fps_sqrt_assign(&mut self, deg: usize) {
        debug_assert!(self.len()>deg);
        *self=FPS::fps_sqrt(self, deg);
    }
    fn fps_sqrt(&self, deg: usize) -> Self {
        debug_assert!(self.len()>deg);
        let mut sqrt=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        let mut lower=0;
        for i in 0..=deg {
            if self[i]!=ac_library::StaticModInt::<M>::new(0) {
                sqrt[0]=self[i].mod_sqrt().unwrap();
                lower=i;
                break;
            }
        }
        debug_assert_eq!(lower%2, 0);
        let mut curlen=1;
        while curlen<=deg {
            curlen*=2;
            let mut g=vec![ac_library::StaticModInt::<M>::new(0);min(curlen,deg+1)];
            for i in 0..curlen/2 {
                g[i]=sqrt[i]*2;
            }
            g=g.fps_inv(min(curlen-1,deg));
            let h=&ac_library::convolution::convolution(&self[lower..min(lower+curlen,deg+1)], &g[0..min(curlen,deg+1)])[curlen/2..min(curlen,deg+1)];
            for i in curlen/2..min(curlen,deg+1) {
                sqrt[i]=h[i-curlen/2];
            }
        }
        for i in (lower/2..=deg).rev() {
            sqrt[i]=sqrt[i-lower/2];
        }
        for i in 0..min(lower/2,deg+1) {
            sqrt[i]=ac_library::StaticModInt::<M>::new(0);
        }
        sqrt
    }
    fn sparse_fps_mul_assign(&mut self, g: &Self::Sparse, deg: usize) {
        self.fps_prefix_assign(deg+1);
        for i in (0..=deg).rev() {
            let tmp=self[i];
            self[i]=ac_library::StaticModInt::<M>::new(0);
            for &(d,c) in g {
                if d==0 {
                    self[i]=self[i]+tmp*c;
                } else if i>=d {
                    self[i]=self[i]+self[i-d]*c;
                }
            }
        }
    }
    fn sparse_fps_mul(f: &Self, g: &Self::Sparse, deg: usize) -> Self {
        let mut h=f.clone();
        h.sparse_fps_mul_assign(g, deg);
        h
    }
    fn sparse_fps_div_assign(&mut self, g: &Self::Sparse, deg: usize) {
        self.fps_prefix_assign(deg+1);
        let mut constant=ac_library::StaticModInt::<M>::new(0);
        for &(d,c) in g {
            if d==0 {
                constant=c;
            }
        }
        debug_assert_ne!(constant, ac_library::StaticModInt::<M>::new(0));
        let cinv=ac_library::StaticModInt::<M>::new(1)/constant;
        for i in 0..=deg {
            for &(d,c) in g {
                if d>0 && i>=d {
                    self[i]=self[i]-self[i-d]*c;
                }
            }
            self[i]*=cinv;
        }
    }
    fn sparse_fps_div(f: &Self, g: &Self::Sparse, deg: usize) -> Self {
        let mut h=f.clone();
        h.sparse_fps_div_assign(g, deg);
        h
    }
    fn fps_prod_merge(fs: &mut Vec<Self>) {
        let len=fs.len();
        let mut pq=RevBinaryHeap::<(usize,usize)>::new();
        for i in 0..len {
            pq.push((fs[i].len()-1,i));
        }
        while let Some((l_deg,mut l_i))=pq.pop() {
            if let Some((r_deg,mut r_i))=pq.pop() {
                if l_i>r_i {
                    std::mem::swap(&mut l_i, &mut r_i);
                }
                let tmp=std::mem::take(&mut fs[r_i]);
                fs[l_i].fps_mul_assign(&tmp, l_deg+r_deg);
                pq.push((l_deg+r_deg,l_i));
            }
        }
    }
}

/// 次数とNTT素数の組のベクターで疎な形式的冪級数を扱うトレイト（通常の形式的冪級数への変換ではdegで結果の次数を指定）
pub trait SparseFPS {
    /// 対応するFPSの型
    type Dense;
    /// 疎な形式的冪級数の逆元を返す関数
    fn sparse_fps_inv(&self, deg: usize) -> Self::Dense;
    /// 疎な形式的冪級数の導関数を割り当てる関数
    fn sparse_fps_diff_assign(&mut self);
    /// 疎な形式的冪級数の導関数を返す関数
    fn sparse_fps_diff(&self) -> Self;
    /// 疎な形式的冪級数の原始関数を割り当てる関数
    fn sparse_fps_int_assign(&mut self);
    /// 疎な形式的冪級数の原始関数を返す関数
    fn sparse_fps_int(&self) -> Self;
    /// 疎な形式的冪級数の対数を返す関数
    fn sparse_fps_log(&self, deg: usize) -> Self::Dense;
    /// 疎な形式的冪級数の指数を返す関数
    fn sparse_fps_exp(&self, deg: usize) -> Self::Dense;
    /// 疎な形式的冪級数の冪を返す関数
    fn sparse_fps_pow(&self, k: usize, deg: usize) -> Self::Dense;
    /// 疎な形式的冪級数の平方根を返す関数
    fn sparse_fps_sqrt(&self, deg: usize) -> Self::Dense;
}

impl<M> SparseFPS for Vec<(usize,ac_library::StaticModInt<M>)> where M: ac_library::Modulus {
    type Dense = Vec<ac_library::StaticModInt<M>>;
    fn sparse_fps_inv(&self, deg: usize) -> Self::Dense {
        let mut ret=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        ret[0]=ac_library::StaticModInt::<M>::new(1);
        ret.sparse_fps_div_assign(self, deg);
        ret
    }
    fn sparse_fps_diff_assign(&mut self) {
        self.retain(|&(d,_)| d>0);
        for (d,c) in self {
            *c*=*d;
            *d-=1;
        }
    }
    fn sparse_fps_diff(&self) -> Self {
        let mut h=self.clone();
        h.sparse_fps_diff_assign();
        h
    }
    fn sparse_fps_int_assign(&mut self) {
        for (d,c) in self {
            *d+=1;
            *c/=*d;
        }
    }
    fn sparse_fps_int(&self) -> Self {
        let mut h=self.clone();
        h.sparse_fps_int_assign();
        h
    }
    fn sparse_fps_log(&self, deg: usize) -> Self::Dense {
        let mut ret=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        for &(d,c) in self {
            ret[d]=c;
        }
        debug_assert_eq!(ret[0], ac_library::StaticModInt::<M>::new(1));
        ret.fps_diff_assign();
        ret.sparse_fps_div_assign(self, deg);
        ret.fps_int_assign();
        ret
    }
    fn sparse_fps_exp(&self, deg: usize) -> Self::Dense {
        let mut ret=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        ret[0]=ac_library::StaticModInt::<M>::new(1);
        for i in 0..deg {
            for &(d,c) in self {
                debug_assert!(d!=0 || c==ac_library::StaticModInt::<M>::new(0));
                if i+1>=d {
                    ret[i+1]=ret[i+1]+ret[i+1-d]*c*d;
                }
            }
            ret[i+1]/=i+1;
        }
        ret
    }
    fn sparse_fps_pow(&self, k: usize, deg: usize) -> Self::Dense {
        let mut lower=(usize::MAX,ac_library::StaticModInt::<M>::new(1));
        for &(d,c) in self {
            if c!=ac_library::StaticModInt::<M>::new(0) && d<=lower.0 {
                lower=(d,c);
            }
        }
        let mut ret=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        ret[0]=ac_library::StaticModInt::<M>::new(1);
        for i in 0..deg {
            for &(d,c) in self {
                let d=d-lower.0;
                let c=c/lower.1;
                if d>0 && i+1>=d {
                    ret[i+1]=ret[i+1]+ret[i+1-d]*c*k*d-ret[i+1-d]*c*(i+1-d);
                }
            }
            ret[i+1]/=i+1;
        }
        let lpow=lower.1.pow(k as u64);
        for i in (lower.0*k..=deg).rev() {
            ret[i]=ret[i-lower.0*k]*lpow;
        }
        for i in 0..min(lower.0*k,deg+1) {
            ret[i]=ac_library::StaticModInt::<M>::new(0);
        }
        ret
    }
    fn sparse_fps_sqrt(&self, deg: usize) -> Self::Dense {
        let mut lower=(usize::MAX,ac_library::StaticModInt::<M>::new(1));
        for &(d,c) in self {
            if c!=ac_library::StaticModInt::<M>::new(0) && d<=lower.0 {
                lower=(d,c);
            }
        }
        let csqrt=lower.1.mod_sqrt().unwrap();
        debug_assert_eq!(lower.0%2, 0);
        let mut ret=vec![ac_library::StaticModInt::<M>::new(0);deg+1];
        ret[0]=ac_library::StaticModInt::<M>::new(1);
        for i in 0..deg {
            for &(d,c) in self {
                let d=d-lower.0;
                let c=c/lower.1;
                if d>0 && i+1>=d {
                    ret[i+1]=ret[i+1]+ret[i+1-d]*c*d/2-ret[i+1-d]*c*(i+1-d);
                }
            }
            ret[i+1]/=i+1;
        }
        for i in (lower.0/2..=deg).rev() {
            ret[i]=ret[i-lower.0/2]*csqrt;
        }
        for i in 0..min(lower.0/2,deg+1) {
            ret[i]=ac_library::StaticModInt::<M>::new(0);
        }
        ret
    }
}

/// Mo's Algorithmの区間の動き方の列挙型（順にクエリの番号と区間の左端と右端）
/// （Fallが直接そのクエリに向かう動き方、Lengthenは区間が長くなる動き方、LShortenは区間が短くなる動き方で返される左端は消える点であることに注意）
pub enum MoS {
    Fall(usize, usize, usize),
    RLengthen(Option<usize>, usize, usize),
    LLengthen(Option<usize>, usize, usize),
    LShorten(Option<usize>, usize, usize)
}

/// Mo's Algorithmの関数（返り値はイテレータ）
pub fn mo_s_algorithm(n: usize, queries: &Vec<(usize,usize)>) -> impl Iterator<Item=MoS> {
    let sqrt=num_integer::Roots::sqrt(&n);
    let mut queries=queries.iter().enumerate().map(|(i,&lr)| (i,lr)).collect::<Vec<_>>();
    queries.sort_by(|&(i1,(l1,r1)),&(i2,(l2,r2))| {
        if l1/sqrt!=l2/sqrt {
            (l1/sqrt).cmp(&(l2/sqrt))
        } else if r1!=r2 {
            r1.cmp(&r2)
        } else {
            i1.cmp(&i2)
        }
    });
    let mut prev_l=n;
    let mut prev_r=n;
    let mut cur_ind=0;
    std::iter::from_fn(move || {
        if cur_ind==queries.len() {
            return None;
        }
        let (i,(l,r))=queries[cur_ind];
        if prev_r>r {
            (prev_l,prev_r)=(l,r);
            cur_ind+=1;
            Some(MoS::Fall(i, l, r))
        } else if prev_r<r {
            prev_r+=1;
            let ret_ind=if prev_l==l && prev_r==r {
                cur_ind+=1;
                Some(i)
            } else {
                None
            };
            Some(MoS::RLengthen(ret_ind, prev_l, prev_r))
        } else if prev_l>l {
            prev_l-=1;
            let ret_ind=if prev_l==l && prev_r==r {
                cur_ind+=1;
                Some(i)
            } else {
                None
            };
            Some(MoS::LLengthen(ret_ind, prev_l, prev_r))
        } else {
            prev_l+=1;
            let ret_ind=if prev_l==l && prev_r==r {
                cur_ind+=1;
                Some(i)
            } else {
                None
            };
            Some(MoS::LShorten(ret_ind, prev_l-1, prev_r))
        }
    })
}

/// プリューファーコードのトレイト
pub trait PrueferCode {
    /// プリューファーコードの表すラベルつき木のVecGraphを返す関数（0-indexed）
    fn labeled_tree_vec(&self) -> VecGraph;
    /// プリューファーコードの表すラベルつき木のMapGraphを返す関数（0-indexed）
    fn labeled_tree_map(&self) -> MapGraph;
}

impl PrueferCode for Vec<usize> {
    fn labeled_tree_vec(&self) -> VecGraph {
        let n=self.len()+2;
        let mut d=vec![1;n];
        let mut leaves=RevBinaryHeap::<usize>::new();
        for &v in self {
            d[v]+=1;
        }
        for v in 0..n {
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let mut ab=Vec::<(usize,usize)>::new();
        for &v in self {
            let u=leaves.pop().unwrap();
            ab.push((v,u));
            d[v]-=1;
            d[u]-=1;
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let v=leaves.pop().unwrap();
        let u=leaves.pop().unwrap();
        ab.push((v,u));
        VecGraph::construct_graph(n, n-1, &ab)
    }
    fn labeled_tree_map(&self) -> MapGraph {
        let n=self.len()+2;
        let mut d=vec![1;n];
        let mut leaves=RevBinaryHeap::<usize>::new();
        for &v in self {
            d[v]+=1;
        }
        for v in 0..n {
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let mut ab=Vec::<(usize,usize)>::new();
        for &v in self {
            let u=leaves.pop().unwrap();
            ab.push((v,u));
            d[v]-=1;
            d[u]-=1;
            if d[v]==1 {
                leaves.push(v);
            }
        }
        let v=leaves.pop().unwrap();
        let u=leaves.pop().unwrap();
        ab.push((v,u));
        MapGraph::construct_graph(n, n-1, &ab)
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

/// 動的セグ木のノードの構造体
#[derive(Clone, Default, Debug)]
struct DynamicSegtreeNode<M> where M: ac_library::Monoid {
    x: M::S,
    left: Option<usize>,
    right: Option<usize>,
}

/// 動的セグ木の構造体
#[derive(Clone, Default, Debug)]
pub struct DynamicSegtree<M> where M: ac_library::Monoid, M::S: std::fmt::Debug {
    nodes: Vec<DynamicSegtreeNode<M>>,
    min_p: isize,
    max_p: isize
}

impl<M> DynamicSegtree<M> where M: ac_library::Monoid, M::S: std::fmt::Debug {
    /// 初期化の関数（min_p、max_pはそれぞれクエリの添字がとる閉区間の下端と上端）
    pub fn new<T>(min_p: T, max_p: T) -> Self where T: num::PrimInt {
        let min_p=min_p.to_isize().unwrap();
        let max_p=max_p.to_isize().unwrap();
        Self { nodes: vec![DynamicSegtreeNode { x: M::identity(), left: None, right: None }], min_p, max_p }
    }
    /// 指定されたノードがあればそのxを、なければ単位元を返す関数
    fn x(&self, node_ind: Option<usize>) -> M::S {
        if let Some(ind)=node_ind {
            self.nodes[ind].x.clone()
        } else {
            M::identity()
        }
    }
    /// 再帰的にxを代入する関数
    fn set_x(&mut self, node_ind: Option<usize>, p: isize, x: M::S, l: isize, r: isize) -> Option<usize> {
        if node_ind.is_none() {
            if l==r {
                self.nodes.push(DynamicSegtreeNode { x, left: None, right: None });
                return Some(self.nodes.len()-1);
            }
            let mut m=(l+r)/2;
            if m==r {
                m-=1;
            }
            if p<=m {
                let left=self.set_x(None, p, x.clone(), l, m);
                self.nodes.push(DynamicSegtreeNode { x: x.clone(), left, right: None });
            } else {
                let right=self.set_x(None, p, x.clone(), m+1, r);
                self.nodes.push(DynamicSegtreeNode { x: x.clone(), left: None, right });
            }
            return Some(self.nodes.len()-1);
        }
        let ind=node_ind.unwrap();
        if l==r {
            self.nodes[ind].x=x;
            return node_ind;
        }
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if p<=m {
            self.nodes[ind].left=self.set_x(self.nodes[ind].left, p, x, l, m);
        } else {
            self.nodes[ind].right=self.set_x(self.nodes[ind].right, p, x, m+1, r);
        }
        self.nodes[ind].x=M::binary_operation(&self.x(self.nodes[ind].left), &self.x(self.nodes[ind].right));
        node_ind
    }
    /// 再帰的にxを返す関数
    fn get_x(&self, node_ind: Option<usize>, p:isize, l: isize, r: isize) -> M::S {
        if node_ind.is_none() {
            return M::identity();
        }
        if l==r {
            return self.x(node_ind).clone();
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if p<=m {
            self.get_x(self.nodes[ind].left, p, l, m)
        } else {
            self.get_x(self.nodes[ind].right, p, m+1, r)
        }
    }
    /// 再帰的に区間のモノイド積を返す関数
    fn get_prod(&self, node_ind: Option<usize>, x_l: isize, x_r: isize, l: isize, r: isize) -> M::S {
        if node_ind.is_none() {
            return M::identity();
        }
        if x_l==l && x_r==r {
            return self.x(node_ind).clone();
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if x_r<=m {
            self.get_prod(self.nodes[ind].left, x_l, x_r, l, m)
        } else if x_l>m {
            self.get_prod(self.nodes[ind].right, x_l, x_r, m+1, r)
        } else {
            M::binary_operation(&self.get_prod(self.nodes[ind].left, x_l, m, l, m), &self.get_prod(self.nodes[ind].right, m+1, x_r, m+1, r))
        }
    }
    /// 再帰的に右端からのモノイド積と条件fを満たさない左端を返す関数
    fn get_right_prod<F>(&self, node_ind: Option<usize>, left: isize, l: isize, r: isize, f: &F) -> (M::S,Option<usize>,isize,isize) where F: Fn(&M::S) -> bool {
        if node_ind.is_none() {
            return (M::identity(),None,l,r);
        }
        if left==l {
            let x=self.x(node_ind);
            return if f(&x) {
                (x.clone(),None,l,r)
            } else {
                (M::identity(),node_ind,l,r)
            };
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if left>m {
            self.get_right_prod(self.nodes[ind].right, left, m+1, r, f)
        } else {
            let (left_x,ind_ret,l_ret,r_ret)=self.get_right_prod(self.nodes[ind].left, left, l, m, f);
            if ind_ret.is_none() {
                let x=M::binary_operation(&left_x, &self.x(self.nodes[ind].right));
                if f(&x) {
                    (x.clone(),None,l,r)
                } else {
                    (left_x,self.nodes[ind].right,m+1,r)
                }
            } else {
                (left_x,ind_ret,l_ret,r_ret)
            }
        }
    }
    /// 再帰的に条件fを満たす右端を返す関数（左閉右開区間）
    fn get_max_right<F>(&self, node_ind: Option<usize>, offset: M::S, l: isize, r: isize, f: &F) -> isize where F: Fn(&M::S) -> bool {
        if node_ind.is_none() {
            return r;
        }
        if l==r {
            let x=M::binary_operation(&offset, &self.x(node_ind));
            return if f(&x) {
                r
            } else {
                l-1
            };
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        let x=M::binary_operation(&offset, &self.x(self.nodes[ind].left));
        if f(&x) {
            self.get_max_right(self.nodes[ind].right, x, m+1, r, f)
        } else {
            self.get_max_right(self.nodes[ind].left, offset, l, m, f)
        }
    }
    /// 再帰的に左端からのモノイド積と条件fを満たさない右端を返す関数
    fn get_left_prod<F>(&self, node_ind: Option<usize>, right: isize, l: isize, r: isize, f: &F) -> (M::S,Option<usize>,isize,isize) where F: Fn(&M::S) -> bool {
        if node_ind.is_none() {
            return (M::identity(),None,l,r);
        }
        if right==r {
            let x=self.x(node_ind);
            return if f(&x) {
                (x.clone(),None,l,r)
            } else {
                (M::identity(),node_ind,l,r)
            };
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if right<=m {
            self.get_left_prod(self.nodes[ind].left, right, l, m, f)
        } else {
            let (right_x,ind_ret,l_ret,r_ret)=self.get_left_prod(self.nodes[ind].right, right, m+1, r, f);
            if ind_ret.is_none() {
                let x=M::binary_operation(&self.x(self.nodes[ind].left), &right_x);
                if f(&x) {
                    (x.clone(),None,l,r)
                } else {
                    (right_x,self.nodes[ind].left,l,m)
                }
            } else {
                (right_x,ind_ret,l_ret,r_ret)
            }
        }
    }
    /// 再帰的に条件fを満たす左端を返す関数（左閉右開区間）
    fn get_min_left<F>(&self, node_ind: Option<usize>, offset: M::S, l: isize, r: isize, f: &F) -> isize where F: Fn(&M::S) -> bool {
        if node_ind.is_none() {
            return l;
        }
        if l==r {
            let x=M::binary_operation(&self.x(node_ind), &offset);
            return if f(&x) {
                l
            } else {
                r+1
            };
        }
        let ind=node_ind.unwrap();
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        let x=M::binary_operation(&self.x(self.nodes[ind].right), &offset);
        if f(&x) {
            self.get_min_left(self.nodes[ind].left, x, l, m, f)
        } else {
            self.get_min_left(self.nodes[ind].right, offset, m+1, r, f)
        }
    }
    /// pの位置にxを代入する関数
    pub fn set<T>(&mut self, p: T, x: M::S) where T: num::PrimInt {
        let p=p.to_isize().unwrap();
        debug_assert!(self.min_p<=p && p<=self.max_p);
        self.set_x(Some(0), p, x, self.min_p, self.max_p);
    }
    /// 単位元を代入することで、実質的にpの位置の値を削除する関数
    pub fn reset<T>(&mut self, p: T) where T: num::PrimInt {
        self.set(p, M::identity());
    }
    /// pの位置のxを返す関数
    pub fn get<T>(&self, p: T) -> M::S where T: num::PrimInt {
        let p=p.to_isize().unwrap();
        debug_assert!(self.min_p<=p && p<=self.max_p);
        self.get_x(Some(0), p, self.min_p, self.max_p)
    }
    /// 区間のモノイド積を返す関数
    pub fn prod<T,R>(&self, range: R) -> M::S where T: num::PrimInt, R: std::ops::RangeBounds<T> {
        if let std::ops::Bound::Excluded(&r)=range.end_bound() {
            let r=r.to_isize().unwrap();
            if r==self.min_p {
                return M::identity();
            }
        }
        if let std::ops::Bound::Excluded(&l)=range.start_bound() {
            let l=l.to_isize().unwrap();
            if l==self.max_p {
                return M::identity();
            }
        }
        let l=match range.start_bound() {
            std::ops::Bound::Included(&l) => l.to_isize().unwrap(),
            std::ops::Bound::Excluded(&l) => l.to_isize().unwrap()+1,
            std::ops::Bound::Unbounded => self.min_p
        };
        let r=match range.end_bound() {
            std::ops::Bound::Included(&r) => r.to_isize().unwrap(),
            std::ops::Bound::Excluded(&r) => r.to_isize().unwrap()-1,
            std::ops::Bound::Unbounded => self.max_p
        };
        debug_assert!(l<=r && self.min_p<=l && r<=self.max_p);
        if l==self.min_p && r==self.max_p {
            self.all_prod()
        } else {
            self.get_prod(Some(0), l, r, self.min_p, self.max_p)
        }
    }
    /// 全区間のモノイド積を返す関数
    pub fn all_prod(&self) -> M::S {
        self.x(Some(0))
    }
    /// 左端を固定したセグ木上の二分探索（fは条件を表す関数）
    pub fn max_right<T,F>(&self, l: T, f: F) -> T where T: num::PrimInt, F: Fn(&M::S) -> bool {
        let l=l.to_isize().unwrap();
        debug_assert!(self.min_p<=l && l<=self.max_p+1);
        debug_assert!(f(&M::identity()));
        if l==self.max_p+1 {
            return T::from(self.max_p+1).unwrap();
        }
        let (offset,node_ind,left,right)=self.get_right_prod(Some(0), l, self.min_p, self.max_p, &f);
        if node_ind.is_none() {
            return T::from(self.max_p+1).unwrap();
        }
        let ret=self.get_max_right(node_ind, offset, left, right, &f);
        T::from(ret+1).unwrap()
    }
    /// 右端を固定したセグ木上の二分探索（fは条件を表す関数）
    pub fn min_left<T,F>(&self, r: T, f: F) -> T where T: num::PrimInt, F: Fn(&M::S) -> bool {
        let r=r.to_isize().unwrap();
        debug_assert!(self.min_p<=r && r<=self.max_p+1);
        debug_assert!(f(&M::identity()));
        if r==self.min_p {
            return T::from(self.min_p).unwrap();
        }
        let r=r-1;
        let (offset,node_ind,left,right)=self.get_left_prod(Some(0), r, self.min_p, self.max_p, &f);
        if node_ind.is_none() {
            return T::from(self.min_p).unwrap();
        }
        let ret=self.get_min_left(node_ind, offset, left, right, &f);
        T::from(ret).unwrap()
    }
}

/// 動的Li Chao Treeのノードの構造体
#[derive(Clone, Default, Debug)]
struct LiChaoTreeNode {
    a: isize,
    b: isize,
    left: Option<usize>,
    right: Option<usize>,
}

/// 動的Li Chao Treeの構造体
#[derive(Clone, Default, Debug)]
pub struct LiChaoTree {
    len: usize,
    nodes: Vec<LiChaoTreeNode>,
    is_min_query: bool,
    min_x: isize,
    max_x: isize
}

impl LiChaoTree {
    /// 初期化の関数（is_min_queryは最小値クエリであるか、min_x、max_xはそれぞれクエリのxがとる閉区間の下端と上端）
    pub fn new<T>(is_min_query: bool, min_x: T, max_x: T) -> Self where T: num::PrimInt {
        let min_x=min_x.to_isize().unwrap();
        let max_x=max_x.to_isize().unwrap();
        Self { len: 0, nodes: vec![LiChaoTreeNode { a: 0, b: isize::MAX, left: None, right: None }], is_min_query, min_x, max_x }
    }
    /// 直線および線分の個数
    pub fn len(&self) -> usize {
        self.len
    }
    /// f(x)=ax+bの値を返す関数
    fn f<T>(&self, ind: usize, x: T) -> T where T: num::PrimInt {
        let x=x.to_isize().unwrap();
        let node=&self.nodes[ind];
        T::from(node.a*x+node.b).unwrap()
    }
    /// 再帰的に直線を挿入する関数
    fn insert_line(&mut self, node_ind: Option<usize>, mut a: isize, mut b: isize, l: isize, r: isize, y_l: isize, y_r: isize) -> Option<usize> {
        if node_ind.is_none() {
            self.nodes.push(LiChaoTreeNode { a, b, left: None, right: None });
            return Some(self.nodes.len()-1);
        }
        let ind=node_ind.unwrap();
        let f_l=self.f(ind, l);
        let f_r=self.f(ind, r);
        if f_l>=y_l && f_r>=y_r {
            self.nodes[ind].a=a;
            self.nodes[ind].b=b;
        } else if !(f_l<=y_l && f_r<=y_r) {
            let mut m=(l+r)/2;
            if m==r {
                m-=1;
            }
            let f_m=self.f(ind, m);
            let y_m=a*m+b;
            if f_m>y_m {
                std::mem::swap(&mut self.nodes[ind].a, &mut a);
                std::mem::swap(&mut self.nodes[ind].b, &mut b);
                if y_l>=f_l {
                    self.nodes[ind].left=self.insert_line(self.nodes[ind].left, a, b, l, m, f_l, f_m);
                } else {
                    self.nodes[ind].right=self.insert_line(self.nodes[ind].right, a, b, m+1, r, f_m+a, f_r);
                }
            } else {
                if f_l>=y_l {
                    self.nodes[ind].left=self.insert_line(self.nodes[ind].left, a, b, l, m, y_l, y_m);
                } else {
                    self.nodes[ind].right=self.insert_line(self.nodes[ind].right, a, b, m+1, r, y_m+a, y_r);
                }
            }
        }
        node_ind
    }
    /// 再帰的に線分を挿入する関数
    fn insert_line_segment(&mut self, mut node_ind: Option<usize>, a: isize, b: isize, l_end: isize, r_end: isize, l: isize, r: isize, y_l: isize, y_r: isize) -> Option<usize> {
        if r<l_end || r_end<l {
            return node_ind;
        }
        if l_end<=l && r<=r_end {
            return self.insert_line(node_ind, a, b, l, r, y_l, y_r);
        }
        if let Some(ind)=node_ind {
            let f_l=self.f(ind, l);
            let f_r=self.f(ind, r);
            if f_l<=y_l && f_r<=y_r {
                return node_ind;
            }
        } else {
            self.nodes.push(LiChaoTreeNode { a: 0, b: isize::MAX, left: None, right: None });
            node_ind=Some(self.nodes.len()-1);
        }
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        let y_m=a*m+b;
        let ind=node_ind.unwrap();
        self.nodes[ind].left=self.insert_line_segment(self.nodes[ind].left, a, b, l_end, r_end, l, m, y_l, y_m);
        self.nodes[ind].right=self.insert_line_segment(self.nodes[ind].right, a, b, l_end, r_end, m+1, r, y_m+a, y_r);
        node_ind
    }
    /// 再帰的に最小値または最大値を返す関数
    fn return_query(&self, node_ind: Option<usize>, x:isize, l: isize, r: isize) -> isize {
        if node_ind.is_none() {
            return isize::MAX;
        }
        let ind=node_ind.unwrap();
        if l==r {
            return self.f(ind, x);
        }
        let mut m=(l+r)/2;
        if m==r {
            m-=1;
        }
        if x<=m {
            min(self.f(ind, x), self.return_query(self.nodes[ind].left, x, l, m))
        } else {
            min(self.f(ind, x), self.return_query(self.nodes[ind].right, x, m+1, r))
        }
    }
    /// 直線を挿入する関数
    pub fn add_line<T>(&mut self, a: T, b: T) where T: num::PrimInt {
        let a=if self.is_min_query {
            a.to_isize().unwrap()
        } else {
            -(a.to_isize().unwrap())
        };
        let b=if self.is_min_query {
            b.to_isize().unwrap()
        } else {
            -(b.to_isize().unwrap())
        };
        self.insert_line(Some(0), a, b, self.min_x, self.max_x, a*self.min_x+b, a*self.max_x+b);
        self.len+=1;
    }
    /// 線分を挿入する関数（l,rは線分をとるxの左閉右開区間）
    pub fn add_line_segment<T1,T2>(&mut self, a: T1, b: T1, l: T2, r: T2) where T1: num::PrimInt, T2: num::PrimInt {
        let a=if self.is_min_query {
            a.to_isize().unwrap()
        } else {
            -(a.to_isize().unwrap())
        };
        let b=if self.is_min_query {
            b.to_isize().unwrap()
        } else {
            -(b.to_isize().unwrap())
        };
        let l=l.to_isize().unwrap();
        let r=r.to_isize().unwrap();
        self.insert_line_segment(Some(0), a, b, l, r-1, self.min_x, self.max_x, a*self.min_x+b, a*self.max_x+b);
        self.len+=1;
    }
    /// 最小値または最大値を返す関数
    pub fn query<T>(&self, x: T) -> T where T: num::PrimInt {
        let x=x.to_isize().unwrap();
        let ret=self.return_query(Some(0), x, self.min_x, self.max_x);
        if self.is_min_query {
            T::from(ret).unwrap()
        } else {
            T::from(-ret).unwrap()
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

impl<T> BitDigits for T where Self: num::PrimInt {
    fn bit_digits(self) -> usize {
        (Self::zero().leading_zeros()-self.leading_zeros()-1) as usize
    }
}

/// 10のi乗のstatic定数
pub static E: [usize;20]=gen_e();

/// 10のi乗のstatic定数を生成するconst関数
const fn gen_e() -> [usize;20] {
    let mut e=[1;20];
    let mut i=1;
    while i<20 {
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

// not_leonian_ac_lib until this line
