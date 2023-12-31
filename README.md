# not_leonian_ac_lib
AtCoderのためのRustの自作ライブラリです。src/lib.rsとsrc/main.rsからなります。

## src/lib.rs
ライブラリです。「cargo doc --open」で実装している内容の詳細を確認できます。

## src/main.rs
main関数です。
Cargo.tomlのdependenciesにこのライブラリを入れ、実際にAtCoderに提出するときは以下の行の代わりにsrc/lib.rsをくっつけて提出すればよいです。
自動化も容易だと思います。
```rust
#[allow(unused_attributes)] #[macro_use] #[allow(unused_imports)] use not_leonian_ac_lib::*;
```

私はコード長を短くするために[not_leonian_ac_submit](https://github.com/NotLeonian/not_leonian_ac_submit)を使用していますが、バグっていてCEになる可能性を完全には否定できないので注意。
