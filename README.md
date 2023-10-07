# not_leonian_ac_lib
AtCoderのためのRustの自作ライブラリです。src/lib.rsとrust.jsonc（とsrc/main.rs）からなります。

## src/lib.rs
ライブラリです。「cargo doc --open」で実装している内容の詳細を確認できます。

## rust.jsonc
関数やトレイトとして実装すると柔軟性がなくなるものについて、VSCodeのユーザスニペットとして実装したものです。

## src/main.rs
main関数です。
Cargo.tomlのdependenciesにこのライブラリを入れ、実際にAtCoderに提出するときは「\#\[allow\(unused\_attributes\)\] \#\[macro\_use\] \#\[allow\(unused\_imports\)\] use not\_leonian\_ac\_lib::\*;」を削除してsrc/lib.rsの後ろに貼り付けて提出すればよいです。
自動化も容易だと思います。
クレート分割した理由としては、結合テストがしやすく保守性が高まることとともに、事前コンパイルができることが挙げられます。
