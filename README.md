# not_leonian_ac_lib
AtCoderのためのRustの自作ライブラリです。src/lib.rsとrust.jsonc（とsrc/main.rs）からなります。
提出した後のコンパイルは結構長いです。

## src/lib.rs
ライブラリです。「cargo doc --open」で実装している内容の詳細を確認できます。

## rust.jsonc
関数やトレイトとして実装すると柔軟性がなくなるものについて、VSCodeのユーザスニペットとして実装したものです。

## src/main.rs
main関数です。
Cargo.tomlのdependenciesにこのライブラリを入れ、実際にAtCoderに提出するときは以下の行を削除してから、src/lib.rsの後ろに貼り付けて提出すればよいです。
自動化も容易だと思います。
```rust
#[allow(unused_attributes)] #[macro_use] #[allow(unused_imports)] use not_leonian_ac_lib::*;
```
