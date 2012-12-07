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
        url: "/load_network_log?id=" + id,
        dataType : "json",
        success: function(data, textStatus) {
            drawChart(data.log);
            var resultsDesc =
                "Training set accuarcy: " + (100 * data.training_set_acc).toFixed(2) + "%<br/>" +
                "Test set accuarcy: " + (100 * data.test_set_acc).toFixed(2) + "%\n";
            $("#networkResults").html(resultsDesc);
        }
    });

    $.ajax({
        url: "/load_network_desc?id=" + id,
        dataType : "json",
        success: function(data, textStatus) {
            $("#networkDescription").text(data.description);
            $("#networkName").text(data.name);
        }
    });
}