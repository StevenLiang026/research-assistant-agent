% FT+R's routing curve on the development set: results/routing-m4.json -> docs/figs/route-curve-m4.png.
% Left: all items, with the threshold scan, random routing at the same share, the oracle, the target of the
% preregistered rule (B2's success minus 2 points) and the frozen threshold. Right: the handwritten items
% only (descriptive). Run from the repository root:
%     matlab -batch "run('scripts/figs/route_curve.m')"
% On 2026-10-17 MATLAB refused to run on the development machine (a license error). The committed figure is
% drawn with TeX Live by scripts/figs/route_curve_tex.py, which follows this design for its upper row and adds
% the row of success against the cost per 100 items that docs/plan.md asks for. This file stays the design of
% the upper row; it has not drawn the committed PNG.

set(groot, 'defaultAxesFontName', 'Microsoft YaHei', 'defaultTextFontName', 'Microsoft YaHei', ...
    'defaultLegendFontName', 'Microsoft YaHei');
r = jsondecode(fileread(fullfile('results', 'routing-m4.json')));
outdir = fullfile('docs', 'figs');
if ~isfolder(outdir)
    mkdir(outdir);
end

fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 11 4.6]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl);
all_items(ax, r);
ax = nexttile(tl);
handwritten(ax, r.handwritten, r.chosen_point.threshold);

exportgraphics(fig, fullfile(outdir, 'route-curve-m4.png'), 'Resolution', 200, 'BackgroundColor', 'white');
close(fig);


function all_items(ax, r)
    points = r.curves.routing;
    oracle = r.curves.oracle;
    hold(ax, 'on');
    plot(ax, 100 * [points.api_share], 100 * [points.success], '-', 'LineWidth', 1.4, ...
        'DisplayName', 'FT+R:扫描阈值');
    plot(ax, 100 * [points.api_share], 100 * [points.random_success], '--', 'LineWidth', 1.2, ...
        'DisplayName', '随机路由(同调用比例)');
    plot(ax, 100 * [oracle.api_share], 100 * [oracle.success], ':', 'LineWidth', 1.4, ...
        'DisplayName', '理想路由');
    target = 100 * (r.b2_success - r.margin);
    yline(ax, target, '-.', sprintf('目标 %.1f%%(B2 − 2 个百分点)', target), 'LabelHorizontalAlignment', ...
        'left', 'HandleVisibility', 'off');
    chosen = r.chosen_point;
    plot(ax, 100 * chosen.api_share, 100 * chosen.success, 'o', 'MarkerSize', 7, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('冻结阈值 %.3f', chosen.threshold));
    text(ax, 100 * chosen.api_share + 2, 100 * chosen.success - 3.5, ...
        sprintf('调用比例 %.1f%%\\newline成功率 %.1f%%', 100 * chosen.api_share, 100 * chosen.success));
    ends(ax, r.ft_success, r.b2_success);
    style(ax, sprintf('全部 %d 道', r.n));
    legend(ax, 'Location', 'southeast', 'Box', 'off');
end


function handwritten(ax, h, threshold)
    points = h.scan;
    hold(ax, 'on');
    plot(ax, 100 * [points.api_share], 100 * [points.success], '-', 'LineWidth', 1.4, ...
        'DisplayName', 'FT+R:扫描阈值');
    plot(ax, 100 * [points.api_share], 100 * [points.random_success], '--', 'LineWidth', 1.2, ...
        'DisplayName', '随机路由(同调用比例)');
    at = h.at_threshold;
    plot(ax, 100 * at.api_share, 100 * at.success, 'o', 'MarkerSize', 7, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('冻结阈值 %.3f 处', threshold));
    text(ax, 100 * at.api_share + 2, 100 * at.success - 3.5, ...
        sprintf('调用比例 %.1f%%\\newline成功率 %.1f%%', 100 * at.api_share, 100 * at.success));
    ends(ax, h.ft_success, h.b2_success);
    style(ax, sprintf('只看手写题 %d 道(描述)', h.n));
    legend(ax, 'Location', 'southeast', 'Box', 'off');
end


function ends(ax, ft, b2)
    text(ax, 1.5, 100 * ft, sprintf('FT %.1f%%', 100 * ft), 'VerticalAlignment', 'bottom');
    text(ax, 98.5, 100 * b2, sprintf('B2 %.1f%%', 100 * b2), 'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right');
end


function style(ax, name)
    set(ax, 'Box', 'off', 'TickDir', 'out');
    xlim(ax, [0 100]);
    ylim(ax, [40 100]);
    grid(ax, 'on');
    xlabel(ax, 'API 调用比例(%)');
    ylabel(ax, '开发集单步成功率(%)');
    title(ax, name, 'FontWeight', 'normal');
end
