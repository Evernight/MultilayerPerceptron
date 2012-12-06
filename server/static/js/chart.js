function drawChart(data) {
    d3.select("#chartContainer").html("");

    var margin = {top: 20, right: 20, bottom: 30, left: 50},
        width = 700 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    var color = d3.scale.category10();

    var x = d3.scale.linear()
        .range([0, width]);

    var y = d3.scale.linear()
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    var lineTrainSet = d3.svg.line()
        .interpolate("basis")
        .x(function(d) { return x(d.t); })
        .y(function(d) { return y(d.train_E); });

    var lineTestSet = d3.svg.line()
        .interpolate("basis")
        .x(function(d) { return x(d.t); })
        .y(function(d) { return y(d.test_E); });

    var svg = d3.select("#chartContainer").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    data.forEach(function(d) {
        d.t = +d.t;
        d.test_E = +d.test_E;
        d.train_E = +d.train_E;
    });

    x.domain(d3.extent(data, function(d) { return d.t; })).nice();
    y.domain(d3.extent(
        data.map(function(d) { return d.train_E; }).concat(data.map(function(d) { return d.test_E; }))
    )).nice();

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
    .append("text")
        .attr("x", width)
        .attr("y", -6)
        .style("text-anchor", "end")
        .text("Epoch");

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
    .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("E value");

    svg.append("path")
        .datum(data)
        .attr("class", "trainSet line")
        .attr("stroke", color(0))
        .attr("d", lineTrainSet);

    svg.append("path")
        .datum(data)
        .attr("class", "testSet line")
        .attr("stroke", color(1))
        .attr("d", lineTestSet);

}