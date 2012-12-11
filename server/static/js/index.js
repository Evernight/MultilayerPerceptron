function makeTabs() {
    var tabContainers = $('div.tabsContainer > div');
    tabContainers.hide().filter(':first').show();
    $('ul.tabNavigation li.tabSelector a').click(function () {
        tabContainers.hide();
        tabContainers.filter(this.hash).show();

        $('ul.tabNavigation li.tabSelector a').removeClass('selected');
        $(this).addClass('selected');

        $('ul.tabNavigation li.tabSelector').removeClass('active');
        $(this).parent().addClass('active');
    }).filter(':first').click();
}

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

    makeTabs()
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

    var loadVisualizer = function() {
        $.ajax({
            url: "/load_data_visualizer?id=" + id,
            dataType : "text",
            success: function(data, textStatus) {
                eval(data);
                $.ajax({
                    url: "/load_data_file?id=" + id,
                    dataType : "text",
                    success: function(data, textStatus) {
                        drawData(data);
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        alert(textStatus + ' ' + errorThrown);
                    }
                });
            },
            error: function(jqXHR, textStatus, errorThrown) {
                alert(textStatus + ' ' + errorThrown);
            }
        });
    };

    $.ajax({
        url: "/load_network_desc?id=" + id,
        dataType : "json",
        success: function(data, textStatus) {
            $("#networkDescription").text(data.description);
            $("#networkName").text(data.name);

            if (data.data_visualizer) {
                loadVisualizer();
                //$('ul.tabNavigation li.tabSelector').addClass('hidden');
            }
            else {
                //$('ul.tabNavigation li.tabSelector').addClass('hidden');
            }
        }
    });
}