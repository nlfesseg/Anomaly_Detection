$(document).ready(function () {
    var config = {};
    var socket = io();
    $("#TrainButton").on('click', train);
    $("#MQTTConnectionButton").on('click', connect_mqtt);
    $("#RunButton").on('click', run);

    socket.on('connect', function (msg) {
    });

    // ----------------------------------------------------- Responses -----------------------------------------------------

    socket.on('my_response', function (msg) {
        log(msg.data, false)
    });

    socket.on('result_response', function (msg) {
        update_charts(msg.data)
    });

    socket.on('correlation_response', function (msg) {
        correlation_heatmap(msg.data, msg.index, msg.columns)
    });

    socket.on('modified_event_response', function () {
        if ($("#InputViewHistory").val() === "") {
            $("#InputViewHistory").val(0);
        }
        if ($("#InputPastHistory").val() === "" || $("#InputPastHistory").val() == 0) {
            $("#InputPastHistory").val(1);
        }
        if ($("#InputFutureTarget").val() === "" || $("#InputFutureTarget").val() == 0) {
            $("#InputFutureTarget").val(1);
        }
        socket.emit('predict', {
            view_history: $("#InputViewHistory").val(),
            past_history: $("#InputPastHistory").val(),
            future_target: $("#InputFutureTarget").val()
        });
    });

    socket.on('csv_response', function () {
        if ($('#SelectRun').children('option').length >= 1) {
            load_run();
        }
        $("#TrainButton").prop('disabled', false);
        $("#RunButton").prop('disabled', false);
        disable_elements(false);
        disable_correlation_button(false);
    });

    socket.on('run_response', function (msg) {
        $('#SelectFeatures option').remove();
        msg.feature_ids.forEach(element => {
            $('#SelectFeatures').append('<option value="' + element + '">' + element + '</option>');
        });
        $('#SelectFeatures').selectpicker('refresh');
        config = msg.config;
        reset_config();
    });

    socket.on('run_ids_response', function (msg) {
        $('#SelectRun option').remove();
        msg.data.forEach(element => {
            $('#SelectRun').append('<option value="' + element + '">' + element + '</option>');
        });
        $('#SelectRun').selectpicker('refresh');
        if ($('#SelectRun').children('option').length >= 1) {
            load_run()
        } else {
            $('#SelectFeatures option').remove();
            $('#SelectFeatures').selectpicker('refresh');
        }
    });

    socket.on('csv_filenames_response', function (msg) {
        msg.data.forEach(element => {
            $('#SelectCSVFile').append('<option value="' + element + '">' + element + '</option>');
        });
        $('#SelectCSVFile').selectpicker('refresh');
    });

    socket.on('train_response', function (msg) {
        $("#TrainButton").prop('disabled', false);
        $("#TrainButton").removeClass('btn-danger').addClass('btn-primary');
        $("#TrainButton").html("Train");
        $("#TrainButton").off('click').on('click', train);

        load_run_ids();
        disable_csv_picker(false);
        $("#LoadButton").prop('disabled', false);

        disable_elements(false);
        $("#RunButton").prop('disabled', false);
    });

    socket.on('training_stopped_response', function () {
        $("#TrainButton").prop('disabled', false);
        $("#TrainButton").removeClass('btn-danger').addClass('btn-primary');
        $("#TrainButton").html("Train");
        $("#TrainButton").off('click').on('click', train);

        disable_csv_picker(true);
        disable_correlation_button(false);
        $("#LoadButton").prop('disabled', false);
        $("#RunButton").prop('disabled', false);
        disable_elements(false);
    });

    socket.on('running_stopped_response', function () {
        $("#RunButton").prop('disabled', false);
        $("#RunButton").removeClass('btn-danger').addClass('btn-success');
        $("#RunButton").html("Run");
        $("#RunButton").off('click').on('click', run);

        disable_csv_picker(false);
        disable_correlation_button(false);
        $("#LoadButton").prop('disabled', false);
        $("#TrainButton").prop('disabled', false);
        disable_elements(false);
    });

    socket.on('mqtt_connection_failed_response', function (msg) {
        $("#MQTTConnectionButton").prop('disabled', false);
        $("#MQTTConnectionButton").html("Connect");
        log(msg.data)
    });

    socket.on('mqtt_connection_success_response', function (msg) {
        $("#MQTTConnectionButton").prop('disabled', false);
        $("#MQTTConnectionButton").removeClass('btn-success').addClass('btn-danger');
        $("#MQTTConnectionButton").html("Disconnect");
        $("#MQTTConnectionButton").off('click').on('click', disconnect_mqtt);
        log(msg.data)
    });

    socket.on('mqtt_disconnect_response', function (msg) {
        $("#MQTTConnectionButton").removeClass('btn-danger').addClass('btn-success');
        $("#MQTTConnectionButton").html("Connect");
        $("#MQTTConnectionButton").off('click').on('click', connect_mqtt);
        log(msg.data)
    });

    // ----------------------------------------------------- Element Events -----------------------------------------------------

    $('#LoadButton').click(function (event) {
        socket.emit('load_csv', {filename: $("#SelectCSVFile").val()})
    });

    $("#SelectRun").change(function (event) {
        load_run()
    });

    $("#ResetButton").click(function (event) {
        reset_config()
    });

    $("#CorrelationButton").click(function (event) {
        socket.emit('load_correlation_matrix')
    });

    $("#DeleteButton").click(function (event) {
        if ($('#SelectRun').children('option').length >= 1) {
            socket.emit('delete_run', {run_id: $("#SelectRun").val()})
        }
    });

    $("input[type='checkbox']").change(function () {
        disable_run_button(true);
        if ($(event.target).is(":checked")) {
            $(event.target).val("1")
        } else {
            $(event.target).val("0")
        }
    });

    // $("input[type='number']").change(function () {
    //     disable_run_button(true);
    // });

    // ----------------------------------------------------- Standard Functions -----------------------------------------------------

    function run() {
        let desired_option = $("#SelectFeatures").val();
        if (desired_option.length > 0) {
            disable_csv_picker(true);
            $("#LoadButton").prop('disabled', true);
            $("#TrainButton").prop('disabled', true);

            disable_elements(true);
            disable_correlation_button(true);

            $("#RunButton").removeClass('btn-success').addClass('btn-danger');
            $("#RunButton").html("Stop");
            $("#RunButton").off('click').on('click', stop_run);
            create_charts($("#SelectFeatures").val());
            socket.emit('run', {
                run_id: $("#SelectRun").val(),
                feature_ids: desired_option,
            })
        }
    }

    function train() {
        disable_csv_picker(true);
        $("#LoadButton").prop('disabled', true);

        disable_elements(true);
        disable_correlation_button(true);
        $("#RunButton").prop('disabled', true);

        $("#TrainButton").removeClass('btn-primary').addClass('btn-danger');
        $("#TrainButton").html("Stop");
        $("#TrainButton").off('click').on('click', stop_train);
        if ($("#InputStepSize").val() === "" || $("#InputStepSize").val() == 0) {
            $("#InputStepSize").val(1);
        }
        socket.emit('train', {
            step_size: $('#InputStepSize').val(),
        })
    }

    function connect_mqtt() {
        $("#MQTTConnectionButton").prop('disabled', true);
        $("#MQTTConnectionButton").html("Connecting...");
        socket.emit('connect_mqttclient', {
            username: $('#InputMQTTUsername').val(),
            password: $('#InputMQTTPassword').val(),
            broker_address: $('#InputMQTTBrokerAddress').val(),
            port: $('#InputMQTTPort').val(),
        });
    }

    function disconnect_mqtt() {
        socket.emit('disconnect_mqttclient');
    }

    function log(text, newline = true) {
        if (newline)
            text = "\n" + text;
        $("#TextAreaError").val($("#TextAreaError").val() + (text));
        $("#TextAreaError").scrollTop($("#TextAreaError")[0].scrollHeight);
    }

    function stop_train() {
        $("#TrainButton").html("Stopping...");
        $("#TrainButton").prop('disabled', true);
        socket.emit('stop_training');
    }

    function stop_run() {
        $("#RunButton").html("Stopping...");
        $("#RunButton").prop('disabled', true);
        socket.emit('stop_running');
    }

    function load_run() {
        socket.emit('load_run', {run_id: $("#SelectRun").val()})
    }

    function load_run_ids() {
        socket.emit('load_run_ids')
    }

    function disable_run_button(disable) {
        $("#RunButton").prop('disabled', disable);
    }

    function disable_correlation_button(disable) {
        $("#CorrelationButton").prop('disabled', disable);
    }

    function disable_csv_picker(disable) {
        $("#SelectCSVFile").prop('disabled', disable);
        $("#SelectCSVFile").selectpicker('refresh');
    }

    function reset_config() {
        $("#InputStepSize").val(config['FORECAST_PARAMS'].step_size);
        disable_run_button(false)
    }

    function disable_elements(disable) {
        $("#SelectFeatures").prop('disabled', disable);
        $("#SelectFeatures").selectpicker('refresh');
        $("#SelectRun").prop('disabled', disable);
        $("#SelectRun").selectpicker('refresh');

        $("#InputPastHistory").prop('disabled', disable);
        $("#InputViewHistory").prop('disabled', disable);
        $("#InputFutureTarget").prop('disabled', disable);
        $("#InputStepSize").prop('disabled', disable);

        $("#ResetButton").prop('disabled', disable);
        $("#DeleteButton").prop('disabled', disable);
    }
});
