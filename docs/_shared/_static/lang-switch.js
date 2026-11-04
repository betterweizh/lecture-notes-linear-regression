// Remember the reader's language choice and keep the scroll position
// (section anchor) when switching between the English and Chinese editions.
(function () {
  "use strict";

  function remember(lang) {
    try {
      window.localStorage.setItem("lnlr-preferred-language", lang);
    } catch (e) {
      /* storage may be unavailable (private mode); ignore */
    }
  }

  document.addEventListener("click", function (event) {
    var link = event.target.closest && event.target.closest(".lang-switch__link");
    if (!link) return;
    remember(link.getAttribute("data-lang"));
    // Section anchors are generated from the heading text, so they differ
    // between editions. Equation and theorem labels are identical in both
    // editions (e.g. #equation-eq-ols, #thm-beta-unbiased), so keep those.
    var hash = window.location.hash;
    if (hash && /^#(equation-eq|eq|thm|lem|def|prop|cor|asm|alg|ex|rem|fig|tab)-/.test(hash)) {
      event.preventDefault();
      window.location.href = link.getAttribute("href") + hash;
    }
  });

  var current = document.documentElement.getAttribute("lang") || "";
  remember(current.toLowerCase().indexOf("zh") === 0 ? "zh" : "en");
})();
