using Plots;
plotly()
vv = 10;
f = plot(data_vintages[1][:,1], data_vintages[1][:,vv]);
for i=2:length(data_vintages)
    plot!(f, data_vintages[i][:,1], data_vintages[i][:,vv]);
end
