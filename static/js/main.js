function showTable(tableId) {
  var tables = document.getElementsByTagName("table");
  for (var i = 0; i < tables.length; i++) {
    tables[i].classList.remove("hidden");
    tables[i].style.marginTop = "20px";
  }

  var table = document.getElementById(tableId);
  if (table) {
    table.classList.remove("hidden");
  }
}
