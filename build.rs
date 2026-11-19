//! Records which Luma this build is compiled against, from `Cargo.lock`, for
//! `lumapy.__luma_version__` / `__luma_commit__` / `__luma_source__`.

use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock = fs::read_to_string("Cargo.lock").unwrap_or_default();

    // The `[[package]]` entry named "luma": its version and, for a git dependency, its source.
    let entry =
        lock.split("[[package]]").find(|block| block.lines().any(|l| l.trim() == "name = \"luma\"")).unwrap_or("");
    let field = |key: &str| {
        entry
            .lines()
            .find_map(|l| l.trim().strip_prefix(key)?.trim().strip_prefix("= ")?.trim_matches('"').to_owned().into())
            .unwrap_or_default()
    };
    let version = field("version");
    let source = field("source");

    // `git+https://github.com/flipnoteman/Luma?rev=<rev>#<commit>`, `registry+...`, or none for a
    // path dependency (a local checkout patched in).
    let (kind, commit) = if let Some(git) = source.strip_prefix("git+") {
        ("git", git.rsplit_once('#').map(|(_, c)| c.to_owned()).unwrap_or_default())
    } else if source.starts_with("registry+") {
        ("crates.io", String::new())
    } else {
        ("local", String::new())
    };

    println!("cargo:rustc-env=LUMAPY_LUMA_VERSION={}", if version.is_empty() { "unknown" } else { &version });
    println!("cargo:rustc-env=LUMAPY_LUMA_COMMIT={commit}");
    println!("cargo:rustc-env=LUMAPY_LUMA_SOURCE={kind}");
}
