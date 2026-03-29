% MATLAB script with GUI to interactively visualize tan(theta) on the unit circle

% Create figure
fig = figure('Name', 'Tangent Visualization', 'NumberTitle', 'off');

% Define the unit circle (static)
theta_circle = linspace(0, 2*pi, 100);
x_circle = cos(theta_circle);
y_circle = sin(theta_circle);
plot(x_circle, y_circle, 'b-', 'LineWidth', 2);
hold on;
axis equal;
axis([-2 2 -2 2]);
grid on;
xlabel('x');
ylabel('y');

% Initial theta
theta_deg = 45;

% Plot initial elements
[point_handle, tangent_handle, arc_handle, theta_text_handle, tan_text_handle] = plotElements(theta_deg);

% Add edit box for theta
edit_box = uicontrol('Style', 'edit', 'String', num2str(theta_deg), ...
    'Position', [150 20 100 20]);

% Add update button
update_button = uicontrol('Style', 'pushbutton', 'String', 'Update', ...
    'Position', [260 20 100 20], 'Callback', @updatePlot);

% Add label for edit box
uicontrol('Style', 'text', 'Position', [150 45 210 20], 'String', 'Theta (degrees)', 'HorizontalAlignment', 'center');

% Title
title_handle = title(['Unit Circle with Tangent at \theta = ', num2str(theta_deg), '^\circ']);

% Store handles in figure's UserData
set(fig, 'UserData', struct('edit_box', edit_box, 'title_handle', title_handle));

hold off;

% Callback function
function updatePlot(src, ~)
    fig = ancestor(src, 'figure');
    data = get(fig, 'UserData');
    theta_str = get(data.edit_box, 'String');
    theta_deg = str2double(theta_str);
    if isnan(theta_deg) || theta_deg < 0 || theta_deg > 360
        msgbox('Please enter a valid angle between 0 and 360 degrees.', 'Invalid Input', 'error');
        return;
    end
    % Update plot elements
    [point_handle, tangent_handle, arc_handle, theta_text_handle, tan_text_handle] = plotElements(theta_deg);
    % Update title
    set(data.title_handle, 'String', ['Unit Circle with Tangent at \theta = ', num2str(theta_deg, '%.1f'), '^\circ']);
end

function [point_h, tangent_h, arc_h, theta_txt_h, tan_txt_h] = plotElements(theta_deg)
    % Convert to radians
    theta_rad = deg2rad(theta_deg);
    
    % Point on the circle
    x_point = cos(theta_rad);
    y_point = sin(theta_rad);
    
    % Slope of the tangent line
    if abs(sin(theta_rad)) > 1e-6
        slope = -x_point / y_point;
    else
        slope = inf;
    end
    
    % Tangent line points
    x_tangent = linspace(-2, 2, 100);
    if isfinite(slope)
        y_tangent = y_point + slope * (x_tangent - x_point);
    else
        y_tangent = linspace(-2, 2, 100);
        x_tangent = x_point * ones(size(y_tangent));
    end
    
    % Angle arc
    theta_arc = linspace(0, theta_rad, 50);
    x_arc = 0.5 * cos(theta_arc);
    y_arc = 0.5 * sin(theta_arc);
    
    % Update or create handles
    persistent p_h t_h a_h tt_h tat_h
    if isempty(p_h) || ~isvalid(p_h)
        p_h = plot(x_point, y_point, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        t_h = plot(x_tangent, y_tangent, 'r--', 'LineWidth', 1.5);
        a_h = plot(x_arc, y_arc, 'g-', 'LineWidth', 1);
        tt_h = text(x_point + 0.1, y_point, ['\theta = ', num2str(theta_deg, '%.1f'), '^\circ']);
        tat_h = text(0.1, 0.9, ['tan(\theta) = ', num2str(tand(theta_deg), '%.3f')]);
    else
        set(p_h, 'XData', x_point, 'YData', y_point);
        set(t_h, 'XData', x_tangent, 'YData', y_tangent);
        set(a_h, 'XData', x_arc, 'YData', y_arc);
        set(tt_h, 'Position', [x_point + 0.1, y_point], 'String', ['\theta = ', num2str(theta_deg, '%.1f'), '^\circ']);
        set(tat_h, 'String', ['tan(\theta) = ', num2str(tand(theta_deg), '%.3f')]);
    end
    point_h = p_h;
    tangent_h = t_h;
    arc_h = a_h;
    theta_txt_h = tt_h;
    tan_txt_h = tat_h;
end
