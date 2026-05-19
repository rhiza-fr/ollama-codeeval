document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('th.sortable').forEach(function(th) {
    th.addEventListener('click', function() {
      var table = th.closest('table');
      var tbody = table.querySelector('tbody');
      if (!tbody) return;
      var idx = Array.from(th.parentNode.children).indexOf(th);
      var rows = Array.from(tbody.querySelectorAll('tr:not(.group-sep)'));
      var dir = th.classList.contains('sort-asc') ? 'desc' : 'asc';
      table.querySelectorAll('th.sortable').forEach(function(h) {
        h.classList.remove('sort-asc', 'sort-desc');
      });
      th.classList.add('sort-' + dir);
      rows.sort(function(a, b) {
        var at = a.children[idx].getAttribute('data-sort-value') || a.children[idx].textContent.trim();
        var bt = b.children[idx].getAttribute('data-sort-value') || b.children[idx].textContent.trim();
        var an = parseFloat(at), bn = parseFloat(bt);
        var cmp;
        if (!isNaN(an) && !isNaN(bn)) {
          cmp = an - bn;
        } else {
          cmp = at.localeCompare(bt);
        }
        return dir === 'asc' ? cmp : -cmp;
      });
      rows.forEach(function(r) { tbody.appendChild(r); });
    });
  });
});
