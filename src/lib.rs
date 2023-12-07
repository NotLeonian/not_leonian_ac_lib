//! <https://github.com/NotLeonian/not_leonian_ac_lib>  
//!   
//! Copyright (c) 2023 Not_Leonian  
//! Released under the MIT license  
//! <https://opensource.org/licenses/mit-license.php>  

/// 1つ以上の値を1行で出力するマクロ
#[macro_export]
macro_rules! outputln {
    ($var:expr) => {
        println!("{}",$var);
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
        println!("{:?}",$var);
    };
    ($var:expr,$($vars:expr),+) => {
        print!("{:?} ",$var);
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
        eprintln!("{}",$var);
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
        eprintln!("{:?}",$var);
    };
    ($var:expr,$($vars:expr),+) => {
        #[cfg(debug_assertions)]
        eprint!("{:?} ",$var);
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

/// usizeにキャストするトレイト
pub trait Usize {
    /// usizeにキャストする関数
    fn usize(self) -> usize;
}

impl<T> Usize for T where Self: num::PrimInt {
    fn usize(self) -> usize {
        self.to_usize().unwrap()
    }
}

/// bool型の値をusizeにキャストする関数
pub fn usize(cond: bool) -> usize {
    cond as usize
}

/// isizeにキャストするトレイト
pub trait Isize {
    /// isizeにキャストする関数
    fn isize(self) -> isize;
}

impl<T> Isize for T where Self: num::PrimInt {
    fn isize(self) -> isize {
        self.to_isize().unwrap()
    }
}

/// bool型の値をisizeにキャストする関数
pub fn isize(cond: bool) -> isize {
    cond as isize
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

/// 2次元ベクターによるグラフの型
pub type VecGraph=Vec<Vec<(usize,usize)>>;
/// BTreeMapのベクターによるグラフの型（隣接の高速な判定が目的の型であるため、多重辺には対応していない）
pub type MapGraph=Vec<std::collections::BTreeMap<usize,usize>>;

/// グラフについてのトレイト ((usize,usize)の2次元ベクターと(usize,usize)のBTreeMapのベクターについて実装)
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
    /// 頂点aから頂点bへの辺があるかどうかを判定し、辺があれば重みを返す関数（返り値はOption）（VecGraphの場合の使用は非推奨）
    fn weight(&self, a: usize, b: usize) -> Option<usize>;
    /// 最短経路の距離を返す関数（is_weightedがtrueでダイクストラ法、falseでBFS）（到達不能ならばusize::MAXが入る）
    fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize>;
    /// グラフからUnion-Find木を構築する関数（0-indexed）
    fn construct_union_find(&self) -> ac_library::Dsu;
    /// グラフが二部グラフであるかを判定し、二部グラフであれば色分けの例を返す関数（返り値はOption）
    fn is_bipartite_graph(&self) -> Option<Vec<bool>>;
    /// 木のプリューファーコードを返す関数（0-indexed）（グラフが無向木でない場合の動作は保証しない）
    fn pruefer_code(&self) -> Vec<usize>;
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
    fn weight(&self, a: usize, b: usize) -> Option<usize> {
        for &(u,w) in &self[a] {
            if u==b {
                return Some(w);
            }
        }
        None
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
    fn pruefer_code(&self) -> Vec<usize> {
        let n=self.size();
        let mut adjacency_list=vec![std::collections::BTreeSet::<usize>::new();n];
        let mut d=vec![0;n];
        for v in 0..n {
            for &(u,_) in &self[v] {
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

impl Graph for MapGraph {
    fn new(n: usize) -> Self {
        vec![std::collections::BTreeMap::<usize,usize>::new();n]
    }
    fn size(&self) -> usize {
        self.len()
    }
    fn push(&mut self, a: usize, b: usize, w: usize) {
        self[a].insert(b,w);
    }
    fn weight(&self, a: usize, b: usize) -> Option<usize> {
        if self[a].contains_key(&b) {
            Some(self[a][&b])
        } else {
            None
        }
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
                for (&u,&w) in &self[v] {
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
                for (&u,&w) in &self[v] {
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
            for (&u,_) in &self[v] {
                uf.merge(v, u);
            }
        }
        uf
    }
    fn is_bipartite_graph(&self) -> Option<Vec<bool>> {
        let mut ts=ac_library::TwoSat::new(self.size());
        for v in 0..self.size() {
            for (&u,_) in &self[v] {
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
    fn pruefer_code(&self) -> Vec<usize> {
        let n=self.size();
        let mut adjacency_list=vec![std::collections::BTreeSet::<usize>::new();n];
        let mut d=vec![0;n];
        for v in 0..n {
            for (&u,_) in &self[v] {
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

/// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
pub fn construct_graph<G>(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> G where G: Graph {
    G::construct_graph(n, m, ab)
}

/// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
pub fn construct_directed_graph<G>(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> G where G: Graph {
    G::construct_directed_graph(n, m, ab)
}

/// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
pub fn construct_weighted_graph<G>(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> G where G: Graph {
    G::construct_weighted_graph(n, m, abw)
}

/// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
pub fn construct_weighted_directed_graph<G>(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> G where G: Graph {
    G::construct_weighted_directed_graph(n, m, abw)
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

/// 二分探索の関数（usize）（left_is_trueはok<badであるかどうか、返り値はOption）
pub fn usize_binary_search<F>(max: usize, left_is_true: bool, determine: F) -> Option<usize> where F: Fn(usize) -> bool {
    if left_is_true {
        let ret=binary_search(-1, max as isize, |mid| {
            determine(mid as usize)
        });
        if ret>=0 {
            Some(ret as usize)
        } else {
            None
        }
    } else {
        let ret=binary_search(max as isize, -1, |mid| {
            determine(mid as usize)
        }) as usize;
        if ret<max {
            Some(ret as usize)
        } else {
            None
        }
    }
}

/// 広義の尺取り法を行う関数（increaseは左側の値に対して右側の値が単調増加であるか、satisfiedは返す境界がdetermineを満たすかどうか）（返り値はイテレータ）
pub fn two_pointers<F>(n: usize, m: usize, increase: bool, satisfied: bool, determine: &F) -> impl Iterator<Item=(usize,Option<usize>)> + '_ where F: Fn(usize,usize) -> bool {
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
                    (l,Some(r as usize))
                } else {
                    (l,None)
                }
            } else {
                if ((r+1) as usize)<m {
                    (l,Some((r+1) as usize))
                } else {
                    (l,None)
                }
            }
        } else {
            while r-1>=0 && determine(l,(r-1) as usize) {
                r-=1;
            }
            if satisfied {
                if (r as usize)<m {
                    (l,Some(r as usize))
                } else {
                    (l,None)
                }
            } else {
                if r-1>=0 {
                    (l,Some((r-1) as usize))
                } else {
                    (l,None)
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
pub fn construct_modint_facts<M>(nmax: usize) -> Vec<M> where M: ModIntFact {
    M::construct_modint_facts(nmax)
}

/// ModIntの階乗の逆元をベクターで列挙する関数
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
    fn new() -> Self;
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
    fn new() -> Self {
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
    fn new() -> Self {
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

/// 座標圧縮のトレイト（0-indexed）
pub trait Compress where Self: Sized {
    /// 圧縮した結果の型
    type T1;
    /// 圧縮する前の値の型
    type T2;
    /// 座標圧縮し、圧縮した結果と圧縮する前の値の一覧を返す関数（0-indexed）
    fn compress(&self) -> (Self::T1, Self::T2);
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

/// 区間のそれぞれの要素にアフィン変換を行う遅延セグ木の構造体
pub struct SegmentAffineTransform<M>(std::marker::PhantomData<M>) where M: ac_library::Monoid;

/// 区間のそれぞれの要素にアフィン変換を行う遅延セグ木のトレイト
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
            if let Some(r)=r {
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
            if let Some(r)=r {
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
            if let Some(r)=r {
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
            if let Some(r)=r {
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

/// ローリングハッシュの剰余の定数
const ROLLING_HASH_MOD:usize=2_305_843_009_213_693_951;

#[derive(Clone, Debug)]
/// ローリングハッシュの基数の構造体
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

#[derive(Clone, Debug)]
/// 累積和によるローリングハッシュの構造体
pub struct RollingHash<const N: usize> {
    hashes: [Vec<usize>;N]
}

impl<const N: usize> RollingHash<N> {
    /// ハッシュの累積和と基数およびその逆元から部分列のローリングハッシュを返す関数（0-indexedの半開区間）
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
    /// ハッシュの累積和と基数およびその逆元から部分列を結合した列のローリングハッシュを返す関数（0-indexedの半開区間）
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
pub trait Sequence<const N: usize> {
    /// 列の中身の型
    type T;
    /// 累積和によるローリングハッシュを返す関数
    fn calculate_rolling_hashes(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N>;
}

impl<const N: usize> Sequence<N> for Vec<char> {
    type T = char;
    fn calculate_rolling_hashes(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
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

impl<const N: usize> Sequence<N> for Vec<usize> {
    type T = usize;
    fn calculate_rolling_hashes(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
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

impl<const N: usize> Sequence<N> for String {
    type T = char;
    fn calculate_rolling_hashes(&self, begin: Self::T, b: &RollingHashBases<N>) -> RollingHash<N> {
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

/// slope trickの構造体（プリミティブな整数型および浮動小数点型に対応）
/// （ただし浮動小数点型で用いる場合は型パラメータをordered_float::OrderedFloatとすることに注意）
#[derive(Clone, Default, Debug)]
pub struct SlopeTrick<T> where T: Ord {
    min: T,
    left_vertices: std::collections::BinaryHeap<T>,
    right_vertices: RevBinaryHeap<T>,
    left_offset: T,
    right_offset: T
}

impl<T> SlopeTrick<ordered_float::OrderedFloat<T>> where T: Default + num::Float + std::ops::AddAssign {
    /// 初期化の関数
    pub fn new() -> Self {
        SlopeTrick::default()
    }
    /// 関数の最小値を返す関数
    pub fn min(&self) -> T {
        self.min.0
    }
    /// 最小値をとる点の最小値を返す関数（存在しなければNoneを返す）
    pub fn left_min_point(&self) -> Option<T> {
        if let Some(&p)=self.left_vertices.peek() {
            Some((p+self.left_offset).0)
        } else {
            None
        }
    }
    /// 最小値をとる点の最大値を返す関数（存在しなければNoneを返す）
    pub fn right_min_point(&self) -> Option<T> {
        if let Some(&p)=self.right_vertices.peek() {
            Some((p+self.right_offset).0)
        } else {
            None
        }
    }
    /// 関数に定数関数を足す関数
    pub fn add_const(&mut self, c: T) {
        self.min+=c;
    }
    /// 関数にmax(p-x,0)を足す関数
    pub fn add_left_slope(&mut self, p: T) {
        let p=ordered_float::OrderedFloat(p);
        if let Some(mp)=self.right_min_point() {
            let mp=ordered_float::OrderedFloat(mp);
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
    /// 関数にmax(x-p,0)を足す関数
    pub fn add_right_slope(&mut self, p: T) {
        let p=ordered_float::OrderedFloat(p);
        if let Some(mp)=self.left_min_point() {
            let mp=ordered_float::OrderedFloat(mp);
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
    /// 形式的冪級数の定積分を割り当てる関数
    fn fps_int_assign(&mut self);
    /// 形式的冪級数の定積分を返す関数
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
    /// 形式的冪級数の配列を受け取ってその総積を計算し、配列を破壊して最初の要素に総積を代入する関数
    fn fps_prod_merge(fs: &mut Vec<Self>);
}

impl<M> FPS for Vec<ac_library::StaticModInt<M>> where M: ac_library::Modulus {
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
            let h=&ac_library::convolution::convolution(&h[curlen/2..curlen], &inv[0..curlen/2])[0..min(deg+1,curlen)-curlen/2];
            for i in curlen/2..min(deg+1,curlen) {
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
            let h=&ac_library::convolution::convolution(&l[curlen/2..curlen], &exp[0..curlen/2])[0..min(deg+1,curlen)-curlen/2];
            for i in curlen/2..min(deg+1,curlen) {
                exp[i]-=h[i-curlen/2];
            }
        }
        exp
    }
    fn fps_pow_assign(&mut self, k: usize, deg: usize) {
        debug_assert!((self.len()-1)*k>=deg);
        let n=self.len()-1;
        self.fps_prefix_assign(deg+1);
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
        self.fps_log_assign(deg);
        self.fps_scalar_assign(k as isize);
        self.fps_exp_assign(deg);
        for i in (lower.1*k..=deg).rev() {
            self[i]=self[i-lower.1*k]*lower.0.pow(k as u64);
        }
        for i in 0..min(lower.1*k,deg+1) {
            self[i]=ac_library::StaticModInt::<M>::new(0);
        }
    }
    fn fps_pow(&self, k: usize, deg: usize) -> Self {
        debug_assert!((self.len()-1)*k>=deg);
        let mut h=self.clone();
        h.fps_pow_assign(k, deg);
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

/// プリューファーコードのトレイト
pub trait PrueferCode {
    /// プリューファーコードの表すラベルつき木を返す関数（0-indexed）
    fn labeled_tree<G>(&self) -> G where G: Graph;
}

impl PrueferCode for Vec<usize> {
    fn labeled_tree<G>(&self) -> G where G: Graph {
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
        construct_graph(n, n-1, &ab)
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

impl<T> BitDigits for T where Self: num::PrimInt {
    fn bit_digits(self) -> usize {
        (Self::zero().leading_zeros()-self.leading_zeros()-1) as usize
    }
}

/// 10のi乗のstatic定数
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

/// isizeとusizeについてimplを定義するマクロ（別のジェネリクスと併用したい場合などに使用）
macro_rules! impl_iusize {
    ($($ty:ty),*) => {
        $(
            impl Compress for Vec<$ty> {
                type T1 = Vec<usize>;
                type T2 = Vec<$ty>;
                fn compress(&self) -> (Self::T1, Self::T2) {
                    let mut list=self.clone();
                    list.sort();
                    list.dedup();
                    let len=list.len();
                    let mut nums=vec![0;self.len()];
                    for i in 0..self.len() {
                        nums[i]=binary_search(0, len, |mid| list[mid]<=self[i]);
                    }
                    (nums,list)
                }
            }

            impl Compress for Vec<Vec<$ty>> {
                type T1 = Vec<Vec<usize>>;
                type T2 = Vec<$ty>;
                fn compress(&self) -> (Self::T1, Self::T2) {
                    let mut list=Vec::new();
                    for i in 0..self.len() {
                        for j in 0..self[i].len() {
                            list.push(self[i][j].clone());
                        }
                    }
                    let len=list.len();
                    let mut nums=vec_range(0, self.len(), |i| vec![0;self[i].len()]);
                    for i in 0..self.len() {
                        for j in 0..self[i].len() {
                            nums[i][j]=binary_search(0, len, |mid| list[mid]<=self[i][j]);
                        }
                    }
                    (nums,list)
                }
            }

            impl SlopeTrick<$ty> {
                pub fn new() -> Self {
                    SlopeTrick::default()
                }
                pub fn min(&self) -> $ty {
                    self.min
                }
                pub fn left_min_point(&self) -> Option<$ty> {
                    if let Some(&p)=self.left_vertices.peek() {
                        Some(p+self.left_offset)
                    } else {
                        None
                    }
                }
                pub fn right_min_point(&self) -> Option<$ty> {
                    if let Some(&p)=self.right_vertices.peek() {
                        Some(p+self.right_offset)
                    } else {
                        None
                    }
                }
                pub fn add_const(&mut self, c: $ty) {
                    self.min+=c;
                }
                pub fn add_left_slope(&mut self, p: $ty) {
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
                pub fn add_right_slope(&mut self, p: $ty) {
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
                pub fn add_abs_slope(&mut self, p: $ty) {
                    self.add_left_slope(p);
                    self.add_right_slope(p);
                }
                pub fn prefix_min(&mut self) {
                    self.right_vertices.clear();
                }
                pub fn suffix_min(&mut self) {
                    self.left_vertices.clear();
                }
                pub fn shift(&mut self, a: $ty) {
                    self.left_offset+=a;
                    self.right_offset+=a;
                }
                pub fn sliding_window_minimum(&mut self, a: $ty, b: $ty) {
                    debug_assert!(a<=b);
                    self.left_offset+=a;
                    self.right_offset+=b;
                }
            }
        )*
    }
}

impl_iusize!(isize, usize);

/// プリミティブな整数型についてimplを定義するマクロ（別のジェネリクスと併用したい場合などに使用）
macro_rules! impl_integer {
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

impl_integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

// not_leonian_ac_lib until this line
