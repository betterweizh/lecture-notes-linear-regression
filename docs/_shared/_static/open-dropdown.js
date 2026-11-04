// Open the folded Proof / Solution blocks that contain the link target, so a
// link to an equation, table or figure inside a folded block shows it.
(function () {
  function openTarget() {
    if (!location.hash) return;
    var id = decodeURIComponent(location.hash.slice(1));
    var el = document.getElementById(id);
    if (!el) return;
    var node = el;
    while (node) {
      if (node.tagName === "DETAILS") node.open = true;
      node = node.parentElement;
    }
    el.scrollIntoView();
  }
  window.addEventListener("load", openTarget);
  window.addEventListener("hashchange", openTarget);
})();
