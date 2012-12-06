$(document).ready(function() {
    $.ajax({
        url: "/networks_list",
        dataType : "json",
        success: function(data, textStatus) {
            var networksList = $("#networksList");
            $.each(data, function(i, value) {
                networksList.append("<option>" + value + "</option>");
            });
            loadNetwork();
        }
    });
});

function loadNetwork() {
    var id = $("#networksList").val();
    $.ajax({
        url: "/load_network?id=" + id,
        dataType : "json",
        success: function(data, textStatus) {
            //$("#networkData").text(data);
            drawChart(data);
        }
    });
}