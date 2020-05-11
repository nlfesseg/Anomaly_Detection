let charts = [];
colours = {"VAR": "#0b63ff", "LSTM": "#fa0100"};

const verticalLinePlugin = {
    getLinePosition: function (chart, pointIndex) {
        const meta = chart.getDatasetMeta(0); // first dataset is used to discover X coordinate of a point
        const data = meta.data;
        return data[pointIndex]._model.x;
    },
    renderVerticalLine: function (chartInstance, pointIndex) {
        const lineLeftOffset = this.getLinePosition(chartInstance, pointIndex);
        const scale = chartInstance.scales['y-axis-0'];
        const context = chartInstance.chart.ctx;

        // render vertical line
        context.beginPath();
        context.strokeStyle = '#ff0000';
        context.moveTo(lineLeftOffset, scale.top);
        context.lineTo(lineLeftOffset, scale.bottom);
        context.stroke();

        // write label
        context.fillStyle = "#ff0000";
        context.textAlign = 'center';
        context.verticalAlign = 'top';
        context.fillText("Now", lineLeftOffset, scale.top - 8);
    },

    afterDatasetsDraw: function (chart, easing) {
        if (chart.config.lineAtIndex) {
            chart.config.lineAtIndex.forEach(pointIndex => this.renderVerticalLine(chart, pointIndex));
        }
    }
};

Chart.plugins.register(verticalLinePlugin);

function create_charts(feature_ids) {
    $('#dashboard').empty();
    charts = [];
    feature_ids.forEach(feature_id => {
        let header = $('<h2/>', {
            class: 'sub-header',
            text: feature_id
        });

        let ctx = $('<canvas style="width: 100%; height: 100%;"/>', {
            id: feature_id,
        });

        let placeholder = $('<div/>', {
            class: 'row placeholder',
        }).append(header, ctx);

        $('#dashboard').append(placeholder);

        charts.push(new Chart(ctx, {
            type: 'line',
            lineAtIndex: [],
            options: {
                responsive: true,
                maintainAspectRatio: true,
                elements: {
                    point: {
                        radius: 0
                    }
                },
                tooltips: {
                    enabled: false
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        filter: function (item, chart) {
                            // Logic to remove a particular legend item goes here
                            return !item.text.includes('Diff');
                        }
                    }
                },
            }
        }));
    });
}

function update_charts(results) {
    for (let i = 0; i < results.length; i++) {
        let result = results[i];
        let chart = charts[i];
        remove_datasets(chart);

        let best_model_index = best_model(result);

        let N = result[best_model_index].history.length + result[best_model_index].forecast.length;
        let labels = Array.from({length: N}, (_, index) => index + 1);
        chart.data.labels = labels;

        let history_data = result[best_model_index].history;
        let history = new Array(N).fill(null);
        history.splice.apply(history, [0, history_data.length].concat(history_data));
        add_dataset(chart, 'History', "#000000", history, []);

        result.forEach(model_result => {
            let expected_data = model_result.expected;
            let forecast_data = model_result.forecast;
            let expected = new Array(N).fill(null);
            expected.splice.apply(expected, [N - (expected_data.length + forecast_data.length), expected_data.length].concat(expected_data));
            add_dataset(chart, model_result.model, colours[model_result.model], expected, [10, 10]);
        });

        let forecast_data = result[best_model_index].forecast;
        let forecast = new Array(N).fill(null);
        forecast.splice.apply(forecast, [N - (forecast_data.length), forecast_data.length].concat(forecast_data));
        add_dataset(chart, 'Forecast', "#00cd28", forecast, []);

        result[best_model_index].past_alerts.forEach(anomaly => {
            add_anomaly_dataset(chart, (anomaly.points + 1), anomaly.value, colours[result[best_model_index].model]);
        });

        result[best_model_index].future_alerts.forEach(anomaly => {
            add_anomaly_dataset(chart, (history_data.length + anomaly.points + 1), anomaly.value, colours[result[best_model_index].model]);
        });

        chart.config.lineAtIndex.push(history_data.length - 1);
        chart.update();
    }
}

function best_model(result) {
    let best_mse = 999;
    let best_model_index = 0;
    for (let i = 0; i < result.length; i++) {
        if (result[i].mse < best_mse) {
            best_mse = result[i].mse;
            best_model_index = i;
        }
    }
    return best_model_index;
}

function add_dataset(chart, label, color, data, dotted) {
    chart.data.datasets.push({
        label: label,
        borderDash: dotted,
        borderColor: color,
        data: data,
        fill: false
    });
}

function add_anomaly_dataset(chart, x, y, colour) {
    chart.data.datasets.push({
        data: [{
            x: x,
            y: y,
            r: 5
        }],
        label: ['Diff'],
        // steppedLine: true,
        backgroundColor: colour,
        type: 'bubble'
    });
}

function remove_datasets(chart) {
    if (chart.data) {
        chart.data.datasets = []
    }
}

function correlation_heatmap(correlation_data, index, columns) {
    let data = [];
    for (let i = 0; i < correlation_data.length; i++) {
        for (let j = 0; j < correlation_data[i].length; j++) {
            let row = {x: index[i], y: columns[j], heat: correlation_data[i][j]};
            data.push(row)
        }
    }
    create_heatmap(data);
}

function create_heatmap(data) {
    $('#dashboard').empty();

    let header = $('<h2/>', {
        class: 'sub-header',
        text: 'Correlation Heatmap'
    });

    let ctx = $('<div id="container" style="width: 100%; height:800px;"/>');

    let placeholder = $('<div/>', {
        class: 'row placeholder',
    }).append(header, ctx);

    $('#dashboard').append(placeholder);

    chart = anychart.heatMap(data);

    var colorScale = anychart.scales.ordinalColor();
    // set range of heat parameter's values and corresponding colors
    colorScale.ranges([
        // set color for all points with the heat parameter less than 1200000
        {less: 0.1, color: "#FFE944"},
        {from: 0.1, to: 0.2, color: "#F9CF3C"},
        {from: 0.2, to: 0.3, color: "#F3B535"},
        {from: 0.3, to: 0.4, color: "#EE9C2D"},
        {from: 0.4, to: 0.5, color: "#E88226"},
        {from: 0.5, to: 0.6, color: "#E2681E"},
        {from: 0.6, to: 0.7, color: "#DC4E17"},
        {from: 0.7, to: 0.8, color: "#D7350F"},
        {from: 0.8, to: 0.9, color: "#D11B08"},
        {from: 0.9, to: 0.99, color: "#CB0100"},
        // set color for all points with the heat parameter more than 3000000
        {greater: 0.99, color: "#515151"}
    ]);

    // apply colorScale for colorizing heat map chart
    chart.colorScale(colorScale);

    chart.xScroller().enabled(true);
    chart.yScroller().enabled(true);

    // set the container id
    chart.container("container");

    // initiate drawing the chart
    chart.draw();
}

