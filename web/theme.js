/* Light/dark theme. Loaded in <head> so the theme is set before the first
   paint (no flash). Follows the OS setting until the user picks one with any
   [data-theme-toggle] button; the choice is remembered in this browser. */
(function () {
  "use strict";
  const KEY = "ss-theme";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const saved = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const apply = (theme) => {
    root.dataset.theme = theme;
    document.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
  };

  apply(saved() || (media.matches ? "dark" : "light"));
  media.addEventListener("change", (e) => { if (!saved()) apply(e.matches ? "dark" : "light"); });

  document.addEventListener("click", (e) => {
    if (!e.target.closest("[data-theme-toggle]")) return;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem(KEY, next); } catch { /* private mode - still switch for this page */ }
    apply(next);
  });
})();
