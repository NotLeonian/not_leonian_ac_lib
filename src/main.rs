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

impl<T, const N: usize> Outputln for [T;N] where T: Sized + std::fmt::Display {
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

impl<T> Outputlines for Vec<T> where T: Outputln {
    fn outputlines(&self) {
        for v in self {
            v.outputln();
        }
    }
}

impl<T, const N: usize> Outputlines for [T;N] where T: Sized + Outputln {
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

/// 存在すれば値を、存在しなければ-1を出力するトレイト
trait OutputIfExists {
    /// 値がmaxより小さければ自身を出力し、maxであれば-1を出力する関数
    fn output_if_exists(self, max: Self);
}

impl OutputIfExists for usize {
    fn output_if_exists(self, max: Self) {
        if self<max {
            println!("{}",self);
        } else {
            println!("-1");
        }
    }
}

/// ベクターの先頭にfilledを追加してmだけ右にずらす関数のトレイト
trait MoveRight where Self: std::ops::Index<usize> {
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
fn vec_range<N,F,T>(begin: N, end: N, func: F) -> Vec<T> where std::ops::Range<N>: Iterator, F: Fn(<std::ops::Range<N> as Iterator>::Item) -> T {
    return (begin..end).map(|i| func(i)).collect::<Vec::<T>>();
}

/// isizeをusizeにキャストするトレイト
trait Usize {
    /// isizeをusizeにキャストする関数
    fn usize(self) -> usize;
}

impl Usize for isize {
    fn usize(self) -> usize {
        self as usize
    }
}

/// usizeをisizeにキャストするトレイト
trait Isize {
    /// usizeをisizeにキャストする関数
    fn isize(self) -> isize;
}

impl Isize for usize {
    fn isize(self) -> isize {
        self as isize
    }
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

impl<T> Chminmax for T where T: Clone + std::cmp::PartialOrd {
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

/// 1次元ベクターについてのchminとchmaxのトレイト
trait ChminmaxVec where Self: std::ops::Index<usize> {
    /// ベクターの中身について、添字challengerの値のほうが大きければ上書きする関数
    fn chmax_vec(&mut self, index: usize, challenger: usize);
    /// ベクターの中身について、添字challengerの値のほうが小さければ上書きする関数
    fn chmin_vec(&mut self, index: usize, challenger: usize);
}

impl<T> ChminmaxVec for Vec<T> where T: Clone + std::cmp::PartialOrd {
    fn chmax_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]>self[index] {
            self[index]=self[challenger].clone();
        }
    }
    fn chmin_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]<self[index] {
            self[index]=self[challenger].clone();
        }
    }
}

impl<T, const N: usize> ChminmaxVec for [T;N] where T: Clone + std::cmp::PartialOrd {
    fn chmax_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]>self[index] {
            self[index]=self[challenger].clone();
        }
    }
    fn chmin_vec(&mut self, index: usize, challenger: usize) {
        if self[challenger]<self[index] {
            self[index]=self[challenger].clone();
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

/// 1次元ベクターについてのAddAssignのトレイト
trait VecAddAssign where Self: std::ops::Index<usize> {
    /// 添字l_indexの値+=添字r_indexの値の関数
    fn vec_add_assign(&mut self, l_index: usize, r_index: usize);
}

impl<T> VecAddAssign for Vec<T> where T: Clone + std::ops::Add<Output=T> {
    fn vec_add_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()+self[r_index].clone();
    }
}

impl<T, const N: usize> VecAddAssign for [T;N] where T: Clone + std::ops::Add<Output=T> {
    fn vec_add_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()+self[r_index].clone();
    }
}

/// 1次元ベクターについてのSubAssignのトレイト
trait VecSubAssign where Self: std::ops::Index<usize> {
    /// 添字l_indexの値-=添字r_indexの値の関数
    fn vec_sub_assign(&mut self, l_index: usize, r_index: usize);
}

impl<T> VecSubAssign for Vec<T> where T: Clone + std::ops::Sub<Output=T> {
    fn vec_sub_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()-self[r_index].clone();
    }
}

impl<T, const N: usize> VecSubAssign for [T;N] where T: Clone + std::ops::Sub<Output=T> {
    fn vec_sub_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()-self[r_index].clone();
    }
}

/// 1次元ベクターについてのMulAssignのトレイト
trait VecMulAssign where Self: std::ops::Index<usize> {
    /// 添字l_indexの値*=添字r_indexの値の関数
    fn vec_mul_assign(&mut self, l_index: usize, r_index: usize);
}

impl<T> VecMulAssign for Vec<T> where T: Clone + std::ops::Mul<Output=T> {
    fn vec_mul_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()*self[r_index].clone();
    }
}

impl<T, const N: usize> VecMulAssign for [T;N] where T: Clone + std::ops::Mul<Output=T> {
    fn vec_mul_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()*self[r_index].clone();
    }
}

/// 1次元ベクターについてのDivAssignのトレイト
trait VecDivAssign where Self: std::ops::Index<usize> {
    /// 添字l_indexの値/=添字r_indexの値の関数
    fn vec_div_assign(&mut self, l_index: usize, r_index: usize);
}

impl<T> VecDivAssign for Vec<T> where T: Clone + std::ops::Div<Output=T> {
    fn vec_div_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()/self[r_index].clone();
    }
}

impl<T, const N: usize> VecDivAssign for [T;N] where T: Clone + std::ops::Div<Output=T> {
    fn vec_div_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()/self[r_index].clone();
    }
}

/// 1次元ベクターについてのRemAssignのトレイト
trait VecRemAssign where Self: std::ops::Index<usize> {
    /// 添字l_indexの値%=添字r_indexの値の関数
    fn vec_rem_assign(&mut self, l_index: usize, r_index: usize);
}

impl<T> VecRemAssign for Vec<T> where T: Clone + std::ops::Rem<Output=T> {
    fn vec_rem_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()%self[r_index].clone();
    }
}

impl<T, const N: usize> VecRemAssign for [T;N] where T: Clone + std::ops::Rem<Output=T> {
    fn vec_rem_assign(&mut self, l_index: usize, r_index: usize) {
        self[l_index]=self[l_index].clone()%self[r_index].clone();
    }
}

/// 2次元ベクターによるグラフの型
#[allow(dead_code)]
type VecGraph=Vec<Vec<(usize,usize)>>;
/// BTreeSetのベクターによるグラフの型
#[allow(dead_code)]
type SetGraph=Vec<std::collections::BTreeSet<(usize,usize)>>;

/// グラフについてのトレイト ((usize,usize)の2次元ベクターと(usize,usize)のBTreeSetのベクターについて実装)
trait Graph where Self: Sized {
    /// グラフを初期化する関数
    fn new(n: usize) -> Self;
    /// 頂点数を返す関数
    fn size(&self) -> usize;
    /// 辺を追加する関数
    fn push(&mut self, a: usize, b: usize, w: usize);
    /// 重みなし無向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
            g.push(b, a, 1);
        }
        return g;
    }
    /// 重みなし有向グラフについて、与えられた頂点数、辺数、辺の一覧から隣接リストを構築する関数（0-indexed）
    fn construct_directed_graph(n: usize, m: usize, ab: &Vec<(usize,usize)>) -> Self {
        assert!(ab.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b) in ab {
            g.push(a, b, 1);
        }
        return g;
    }
    /// 重みつき無向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
            g.push(b, a, w);
        }
        return g;
    }
    /// 重みつき有向グラフについて、与えられた頂点数、辺数、辺と重みの一覧から隣接リストを構築する関数（0-indexed）
    fn construct_weighted_directed_graph(n: usize, m: usize, abw: &Vec<(usize,usize,usize)>) -> Self {
        assert!(abw.len()==m);
        let mut g: Self=Graph::new(n);
        for &(a,b,w) in abw {
            g.push(a, b, w);
        }
        return g;
    }
    /// 最短経路の距離を返す関数（is_weightedがtrueでダイクストラ法、falseでBFS）
    fn dist_of_shortest_paths(&self, start: usize, is_weighted: bool) -> Vec<usize>;
}

impl Graph for Vec<Vec<(usize,usize)>> {
    fn new(n: usize) -> Self {
        return vec![Vec::<(usize,usize)>::new();n];
    }
    fn size(&self) -> usize {
        return self.len();
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
                    assert!(w==1);
                    if !seen[u] {
                        dist[u]=dist[v]+w;
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        return dist;
    }
}

impl Graph for Vec<std::collections::BTreeSet<(usize,usize)>> {
    fn new(n: usize) -> Self {
        return vec![std::collections::BTreeSet::<(usize,usize)>::new();n];
    }
    fn size(&self) -> usize {
        return self.len();
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
                    assert!(w==1);
                    if !seen[u] {
                        dist[u]=dist[v]+w;
                        seen[u]=true;
                        queue.push_back(u);
                    }
                }
            }
        }
        return dist;
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

/// ModIntの逆元についてのトレイト
trait ModIntInv where Self: Sized {
    /// 1からnについてのModIntでの逆元をベクターで列挙する関数（最初の要素には0が入る）
    fn construct_modint_inverses(n: usize) -> Vec<Self>;
}

impl<M> ModIntInv for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn construct_modint_inverses(n: usize) -> Vec<Self> {
        assert!(M::HINT_VALUE_IS_PRIME);
        let mut inv=vec![Self::raw(1);n+1];
        for i in 2..=n {
            inv[i]=-inv[Self::modulus() as usize%i]*(Self::modulus() as usize/i);
        }
        inv[0]=Self::raw(0);
        return inv;
    }
}

impl<I> ModIntInv for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn construct_modint_inverses(n: usize) -> Vec<Self> {
        let mut invs=vec![Self::raw(1);n+1];
        for i in 2..=n {
            assert!(Self::modulus() as usize%i > 0);
            invs[i]=-invs[Self::modulus() as usize%i]*(Self::modulus() as usize/i);
        }
        return invs;
    }
}

/// ModIntの階乗についてのトレイト
trait ModIntFact where Self: Sized {
    /// 1からnについてのModIntでの階乗をベクターで列挙する関数
    fn construct_modint_facts(n: usize) -> Vec<Self>;
    /// 1からnについてのModIntでの階乗の逆元をベクターで列挙する関数
    fn construct_modint_fact_inverses(n: usize, invs: &Vec<Self>) -> Vec<Self>;
}

impl<M> ModIntFact for ac_library::StaticModInt<M> where M: ac_library::Modulus {
    fn construct_modint_facts(n: usize) -> Vec<Self> {
        let mut facts=vec![Self::raw(1);n+1];
        for i in 2..=n {
            facts[i]=facts[i-1]*i;
        }
        return facts;
    }
    fn construct_modint_fact_inverses(n: usize, invs: &Vec<Self>) -> Vec<Self> {
        assert!(invs.len() > n);
        let mut factinvs=vec![Self::raw(1);n+1];
        for i in 2..=n {
            factinvs[i]=factinvs[i-1]*invs[i];
        }
        return factinvs;
    }
}

impl<I> ModIntFact for ac_library::DynamicModInt<I> where I: ac_library::Id {
    fn construct_modint_facts(n: usize) -> Vec<Self> {
        let mut facts=vec![Self::raw(1);n+1];
        for i in 2..=n {
            facts[i]=facts[i-1]*i;
        }
        return facts;
    }
    fn construct_modint_fact_inverses(n: usize, invs: &Vec<Self>) -> Vec<Self> {
        assert!(invs.len() > n);
        let mut factinvs=vec![Self::raw(1);n+1];
        for i in 2..=n {
            factinvs[i]=factinvs[i-1]*invs[i];
        }
        return factinvs;
    }
}

/// 累積和についてのトレイト
trait PrefixSum {
    /// 累積和のベクターを構築する関数
    fn construct_prefix_sum(array: &Self) -> Self;
    /// 構築した累積和のベクターから部分和を計算する関数（0-indexedの左閉右開区間）
    fn calculate_partial_sum(&self, l: usize, r: usize) -> Self::Output where Self: std::ops::Index<usize>;
}

impl<T> PrefixSum for Vec<T> where T: Clone + std::ops::Add<Output=T> + std::ops::Sub<Output=T> {
    fn construct_prefix_sum(array: &Self) -> Self {
        let mut prefix_sum=vec![array[0].clone()-array[0].clone();array.len()+1];
        for i in 0..array.len() {
            prefix_sum[i+1]=prefix_sum[i].clone()+array[i].clone();
        }
        return prefix_sum;
    }
    fn calculate_partial_sum(&self, l: usize, r: usize) -> <Self as std::ops::Index<usize>>::Output {
        assert!(l < self.len() && r <= self.len());
        return self[r].clone()-self[l].clone();
    }
}

/// 2次元累積和についてのトレイト
trait TwoDPrefixSum {
    /// 2次元累積和のベクターを構築する関数
    fn construct_2d_prefix_sum(array: &Self) -> Self;
    /// 構築した2次元累積和のベクターから部分和を計算する関数（0-indexedの左閉右開区間）
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <Self::Output as std::ops::Index<usize>>::Output where Self: std::ops::Index<usize>, Self::Output: std::ops::Index<usize>;
}

impl<T> TwoDPrefixSum for Vec<Vec<T>> where T: Clone + std::ops::Add<Output=T> + std::ops::Sub<Output=T> {
    fn construct_2d_prefix_sum(array: &Self) -> Self {
        let mut prefix_sum=vec![vec![array[0][0].clone()-array[0][0].clone();array[0].len()+1];array.len()+1];
        for i in 0..array.len() {
            assert!(array[i].len()==array[0].len());
            for j in 0..array[0].len() {
                prefix_sum[i+1][j+1]=prefix_sum[i+1][j].clone()+array[i][j].clone();
            }
        }
        for j in 0..array[0].len() {
            for i in 0..array.len() {
                prefix_sum[i+1][j+1]=prefix_sum[i][j+1].clone()+prefix_sum[i+1][j+1].clone();
            }
        }
        return prefix_sum;
    }
    fn calculate_2d_partial_sum(&self, l_i: usize, l_j: usize, r_i: usize, r_j: usize) -> <<Self as std::ops::Index<usize>>::Output as std::ops::Index<usize>>::Output {
        assert!(l_i < self.len() && l_j < self[0].len() && r_i <= self.len() && r_j <= self[0].len());
        return self[r_i][r_j].clone()-self[r_i][l_j].clone()-self[l_i][r_j].clone()+self[l_i][l_j].clone();
    }
}

// 素数に関するトレイト
trait Primes where Self: Sized {
    // 素数か判定する関数
    fn is_prime(self) -> bool;
    // 素数冪か判定する関数
    fn is_prime_power(self) -> bool;
    // 約数を列挙する関数
    fn enumerate_divisors(self) -> Vec<Self>;
    // 素因数分解をする関数
    fn prime_factorize(self) -> Vec<(Self,Self)>;
    // ルジャンドルの定理でselfの階乗がpで何回割り切れるかを計算する関数
    fn legendre_s_formula(self, p: usize) -> usize;
    // エラトステネスの篩で素数を列挙する関数
    fn sieve_of_eratosthenes(nmax: Self) -> Vec<bool>;
    // 線形篩で最小素因数を列挙する関数
    fn linear_sieve(nmax: Self) -> Vec<Self>;
    // 線形篩を用いて素因数分解をする関数
    fn fast_prime_factorize(self, linear_sieve: &Vec<Self>) -> Vec<(Self,Self)>;
}

impl Primes for usize {
    fn is_prime(self) -> bool {
        for i in 2..=num_integer::sqrt(self) {
            if self%i==0 {
                return false;
            }
        }
        return true;
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
        return true;
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
        return divs;
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
        return pes;
    }
    fn legendre_s_formula(mut self, p: usize) -> usize {
        let mut e=0;
        while self>0 {
            e+=self/p;
            self/=p;
        }
        return e;
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
        return is_prime;
    }
    fn linear_sieve(nmax: Self) -> Vec<Self> {
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
        return lpf;
    }
    fn fast_prime_factorize(mut self, linear_sieve: &Vec<Self>) -> Vec<(Self,Self)> {
        assert!(linear_sieve.len() > self);
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
        return pes;
    }
}

/// NTT素数のベクターで形式的冪級数を扱うトレイト
trait FPS {
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
    fn fps_scalar(f: &Self, k: isize) -> Self;
    /// 形式的冪級数の積を割り当てる関数
    fn fps_mul_assign(&mut self, g: &Self);
    /// 形式的冪級数の積を返す関数
    fn fps_mul(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の逆元を返す関数
    fn fps_inv(&self) -> Self;
    /// 形式的冪級数の商を割り当てる関数
    fn fps_div_assign(&mut self, g: &Self);
    /// 形式的冪級数の商を返す関数
    fn fps_div(f: &Self, g: &Self) -> Self;
    /// 形式的冪級数の導関数を割り当てる関数
    fn fps_diff_assign(&mut self);
    /// 形式的冪級数の導関数を返す関数
    fn fps_diff(f: &Self) -> Self;
    /// 形式的冪級数の定積分を割り当てる関数
    fn fps_int_assign(&mut self);
    /// 形式的冪級数の定積分を返す関数
    fn fps_int(f: &Self) -> Self;
    /// 形式的冪級数の対数を割り当てる関数
    fn fps_log_assign(&mut self);
    /// 形式的冪級数の対数を返す関数
    fn fps_log(f: &Self) -> Self;
    /// 形式的冪級数の指数を割り当てる関数
    fn fps_exp_assign(&mut self);
    /// 形式的冪級数の指数を返す関数
    fn fps_exp(f: &Self) -> Self;
    /// 形式的冪級数の冪を割り当てる関数
    fn fps_pow_assign(&mut self, k: usize);
    /// 形式的冪級数の冪を返す関数
    fn fps_pow(f: &Self, k: usize) -> Self;
}

impl<M> FPS for Vec<ac_library::StaticModInt<M>> where M: ac_library::Modulus {
    fn fps_add_assign(&mut self, g: &Self) {
        assert!(self.len() == g.len());
        let n=self.len()-1;
        for i in 0..=n {
            self[i]+=g[i];
        }
    }
    fn fps_add(f: &Self, g: &Self) -> Self {
        assert!(f.len() == g.len());
        let mut h=f.clone();
        h.fps_add_assign(&g);
        return h;
    }
    fn fps_sub_assign(&mut self, g: &Self) {
        assert!(self.len() == g.len());
        let n=self.len()-1;
        for i in 0..=n {
            self[i]-=g[i];
        }
    }
    fn fps_sub(f: &Self, g: &Self) -> Self {
        assert!(f.len() == g.len());
        let mut h=f.clone();
        h.fps_sub_assign(&g);
        return h;
    }
    fn fps_scalar_assign(&mut self, k: isize) {
        let n=self.len()-1;
        for i in 0..=n {
            self[i]*=k;
        }
    }
    fn fps_scalar(f: &Self, k: isize) -> Self {
        let mut h=f.clone();
        h.fps_scalar_assign(k);
        return h;
    }
    fn fps_mul_assign(&mut self, g: &Self) {
        assert!(self.len() == g.len());
        let n=self.len()-1;
        let h=FPS::fps_mul(self, g);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn fps_mul(f: &Self, g: &Self) -> Self {
        assert!(f.len() == g.len());
        let n=f.len()-1;
        return ac_library::convolution::convolution(&f[0..=n], &g[0..=n])[0..=n].to_vec();
    }
    fn fps_inv(&self) -> Self {
        let n=self.len()-1;
        let mut inv=vec![ac_library::StaticModInt::<M>::raw(0);n+1];
        inv[0]=self[0].inv();
        let mut curdeg=1;
        while curdeg<=n {
            curdeg*=2;
            let mut f=self[0..std::cmp::min(curdeg,n+1)].to_vec();
            let mut g=vec![ac_library::StaticModInt::<M>::raw(0);std::cmp::min(curdeg,n+1)];
            for i in 0..curdeg/2 {
                g[i]=inv[i];
            }
            f.fps_mul_assign(&g);
            f.fps_mul_assign(&g);
            for i in curdeg/2..std::cmp::min(curdeg,n+1) {
                inv[i]-=f[i];
            }
        }
        return inv;
    }
    fn fps_div_assign(&mut self, g: &Self) {
        assert!(self.len() == g.len());
        self.fps_mul_assign(&g.fps_inv());
    }
    fn fps_div(f: &Self, g: &Self) -> Self {
        assert!(f.len() == g.len());
        let mut h=f.clone();
        h.fps_div_assign(&g);
        return h;
    }
    fn fps_diff_assign(&mut self) {
        let n=self.len()-1;
        for i in 0..n {
            self[i]=self[i+1]*(i+1);
        }
    }
    fn fps_diff(f: &Self) -> Self {
        let mut h=f.clone();
        h.fps_diff_assign();
        return h;
    }
    fn fps_int_assign(&mut self) {
        let n=self.len()-1;
        for i in (1..=n).rev() {
            self[i]=self[i-1]/i;
        }
        self[0]=ac_library::StaticModInt::<M>::raw(0);
    }
    fn fps_int(f: &Self) -> Self {
        let mut h=f.clone();
        h.fps_int_assign();
        return h;
    }
    fn fps_log_assign(&mut self) {
        let n=self.len()-1;
        let h=FPS::fps_log(self);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn fps_log(f: &Self) -> Self {
        let mut h=f.clone();
        h.fps_diff_assign();
        h.fps_div_assign(f);
        h.fps_int_assign();
        return h;
    }
    fn fps_exp_assign(&mut self) {
        let n=self.len()-1;
        let h=FPS::fps_exp(self);
        for i in 0..=n {
            self[i]=h[i];
        }
    }
    fn fps_exp(f: &Self) -> Self {
        assert!(f[0] == ac_library::StaticModInt::<M>::raw(0));
        let n=f.len()-1;
        let mut exp=vec![ac_library::StaticModInt::<M>::raw(0);n+1];
        exp[0]=ac_library::StaticModInt::<M>::raw(1);
        let mut curdeg=1;
        while curdeg<=n {
            curdeg*=2;
            let mut fc=vec![ac_library::StaticModInt::<M>::raw(0);n+1];
            for i in 0..std::cmp::min(curdeg,n+1) {
                fc[i]=f[i];
            }
            fc.fps_sub_assign(&FPS::fps_log(&exp));
            fc[0]+=1;
            exp.fps_mul_assign(&fc);
        }
        return exp;
    }
    fn fps_pow_assign(&mut self, k: usize) {
        let n=self.len()-1;
        let mut lower=(ac_library::StaticModInt::<M>::raw(1),0);
        for i in 0..=n {
            if self[i]!=ac_library::StaticModInt::<M>::raw(0) {
                lower=(self[i],i);
                break;
            }
        }
        for i in 0..=n-lower.1 {
            self[i]=self[i+lower.1]/lower.0;
        }
        for i in n-lower.1+1..=n {
            self[i]=ac_library::StaticModInt::<M>::raw(0);
        }
        self.fps_log_assign();
        self.fps_scalar_assign(k as isize);
        self.fps_exp_assign();
        for i in (lower.1*k..=n).rev() {
            self[i]=self[i-lower.1*k]*lower.0.pow(k as u64);
        }
        for i in 0..std::cmp::min(lower.1*k,n+1) {
            self[i]=ac_library::StaticModInt::<M>::raw(0);
        }
    }
    fn fps_pow(f: &Self, k: usize) -> Self {
        let mut h=f.clone();
        h.fps_pow_assign(k);
        return h;
    }
}

/// 最小値を取り出すことのできる優先度つきキューの構造体
#[allow(dead_code)]
#[derive(Clone, Default, std::fmt::Debug)]
struct RevBinaryHeap<T> where T: Ord {
    binary_heap: std::collections::BinaryHeap<std::cmp::Reverse<T>>
}

impl<T> RevBinaryHeap<T> where T: Ord {
    #[allow(dead_code)]
    fn new() -> RevBinaryHeap<T> {
        return RevBinaryHeap { binary_heap: std::collections::BinaryHeap::<std::cmp::Reverse<T>>::new() };
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

/// 高速ゼータ変換および高速メビウス変換についてのトレイト
trait ZetaMobius {
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

/// 2進法での桁数を求めるトレイト
trait Blen {
    /// 2進法での桁数を求める関数
    fn blen(self) -> usize;
}

impl Blen for usize {
    fn blen(self) -> usize {
        return self.ilog2() as usize;
    }
}
