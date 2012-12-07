function drawData(rawData) {
    var dataLines = rawData.split('\n').slice(1);
    var data = dataLines.map(function (line) {
       var vals = line.split('\t');
       return {
           "x" : vals[0],
           "y" : vals[1]
       }
    });
    window.data11 = data;

    var container = d3.select("#dataChartContainer");
    container.html("");


}

window.drawData = drawData;