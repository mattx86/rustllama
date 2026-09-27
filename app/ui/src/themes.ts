// Bundled highlight.js theme CSS strings. The Settings page's
// `ui.code_theme` field picks one of these; App.tsx injects the
// matching block into a <head><style id="hljs-theme-active"> element.
//
// Each entry overrides the same selectors as the static github-dark
// block in index.html, so the injected stylesheet wins on equal-
// specificity tie-breaking (it lives later in the DOM than the
// inline <style>). That keeps the fallback path safe — if a user
// configures a theme name we don't ship, the static github-dark
// renders without modification.
//
// Themes are abridged from the canonical highlight.js distribution
// at https://github.com/highlightjs/highlight.js/tree/main/src/styles
// — only the color-mapping rules survived; size/padding rules already
// live in index.html and don't change per theme.

export interface CodeTheme {
  name: string;
  label: string;
  /// Background color of the page-level `pre` block (passed to the
  /// chat-page `<pre>` style so the surrounding container matches).
  preBg: string;
  /// CSS string applied to `<head>` when this theme is selected.
  css: string;
}

const githubDark: CodeTheme = {
  name: "github-dark",
  label: "GitHub Dark (default)",
  preBg: "#161b22",
  // Empty string — the static block in index.html IS github-dark, so
  // we don't need to inject anything. Picking this option just clears
  // the injected style and lets the static one show through.
  css: "",
};

const monokai: CodeTheme = {
  name: "monokai",
  label: "Monokai",
  preBg: "#272822",
  css: `
.markdown pre { background: #272822; border-color: #3e3d32 !important; }
.hljs { color: #ddd; background: #272822 }
.hljs-tag, .hljs-keyword, .hljs-selector-tag, .hljs-literal,
.hljs-strong, .hljs-name { color: #f92672 }
.hljs-code { color: #66d9ef }
.hljs-class .hljs-title { color: #fff }
.hljs-attribute, .hljs-symbol, .hljs-regexp, .hljs-link { color: #bf79db }
.hljs-string, .hljs-bullet, .hljs-subst, .hljs-title, .hljs-section,
.hljs-emphasis, .hljs-type, .hljs-built_in, .hljs-builtin-name,
.hljs-selector-attr, .hljs-selector-pseudo, .hljs-addition,
.hljs-variable, .hljs-template-tag, .hljs-template-variable { color: #a6e22e }
.hljs-comment, .hljs-quote, .hljs-deletion, .hljs-meta { color: #75715e }
.hljs-keyword, .hljs-selector-tag, .hljs-literal, .hljs-doctag,
.hljs-title, .hljs-section, .hljs-type, .hljs-selector-id { font-weight: 700 }
`.trim(),
};

const atomOneDark: CodeTheme = {
  name: "atom-one-dark",
  label: "Atom One Dark",
  preBg: "#282c34",
  css: `
.markdown pre { background: #282c34; border-color: #3e4451 !important; }
.hljs { color: #abb2bf; background: #282c34 }
.hljs-comment, .hljs-quote { color: #5c6370; font-style: italic }
.hljs-doctag, .hljs-keyword, .hljs-formula { color: #c678dd }
.hljs-section, .hljs-name, .hljs-selector-tag, .hljs-deletion,
.hljs-subst { color: #e06c75 }
.hljs-literal { color: #56b6c2 }
.hljs-string, .hljs-regexp, .hljs-addition, .hljs-attribute,
.hljs-meta .hljs-string { color: #98c379 }
.hljs-attr, .hljs-variable, .hljs-template-variable, .hljs-type,
.hljs-selector-class, .hljs-selector-attr, .hljs-selector-pseudo,
.hljs-number { color: #d19a66 }
.hljs-symbol, .hljs-bullet, .hljs-link, .hljs-meta, .hljs-selector-id,
.hljs-title { color: #61aeee }
.hljs-built_in, .hljs-title.class_, .hljs-class .hljs-title { color: #e6c07b }
.hljs-emphasis { font-style: italic }
.hljs-strong { font-weight: 700 }
`.trim(),
};

const dracula: CodeTheme = {
  name: "dracula",
  label: "Dracula",
  preBg: "#282a36",
  css: `
.markdown pre { background: #282a36; border-color: #44475a !important; }
.hljs { color: #f8f8f2; background: #282a36 }
.hljs-built_in, .hljs-selector-tag, .hljs-section, .hljs-tag { color: #8be9fd }
.hljs-keyword { color: #ff79c6 }
.hljs-subst, .hljs-attr, .hljs-attribute, .hljs-literal,
.hljs-meta, .hljs-name, .hljs-selector-id, .hljs-selector-class,
.hljs-type, .hljs-variable { color: #50fa7b }
.hljs-string, .hljs-regexp, .hljs-symbol { color: #f1fa8c }
.hljs-bullet, .hljs-link, .hljs-number, .hljs-quote, .hljs-emphasis,
.hljs-strong, .hljs-title, .hljs-section { color: #bd93f9 }
.hljs-comment, .hljs-code, .hljs-formula { color: #6272a4 }
.hljs-emphasis { font-style: italic }
.hljs-strong { font-weight: 700 }
`.trim(),
};

const solarizedDark: CodeTheme = {
  name: "solarized-dark",
  label: "Solarized Dark",
  preBg: "#002b36",
  css: `
.markdown pre { background: #002b36; border-color: #073642 !important; }
.hljs { color: #839496; background: #002b36 }
.hljs-comment, .hljs-quote { color: #586e75 }
.hljs-keyword, .hljs-selector-tag, .hljs-addition { color: #859900 }
.hljs-number, .hljs-string, .hljs-meta .hljs-meta-string, .hljs-literal,
.hljs-doctag, .hljs-regexp { color: #2aa198 }
.hljs-title, .hljs-section, .hljs-name, .hljs-selector-id,
.hljs-selector-class { color: #268bd2 }
.hljs-attribute, .hljs-attr, .hljs-variable, .hljs-template-variable,
.hljs-class .hljs-title, .hljs-type { color: #b58900 }
.hljs-symbol, .hljs-bullet, .hljs-subst, .hljs-meta, .hljs-meta .hljs-keyword,
.hljs-selector-attr, .hljs-selector-pseudo, .hljs-link { color: #cb4b16 }
.hljs-built_in, .hljs-deletion { color: #dc322f }
.hljs-formula { background: #073642 }
.hljs-emphasis { font-style: italic }
.hljs-strong { font-weight: 700 }
`.trim(),
};

export const CODE_THEMES: CodeTheme[] = [
  githubDark,
  monokai,
  atomOneDark,
  dracula,
  solarizedDark,
];

/// Find a theme by `name`. Falls back to github-dark for unknown
/// values so a config typo doesn't break the chat rendering.
export function findCodeTheme(name: string): CodeTheme {
  return CODE_THEMES.find((t) => t.name === name) ?? githubDark;
}
