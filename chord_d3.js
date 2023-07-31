// Chordplot based on the incoming connections i.e. how many elements decompose into a particular one.

let jsonData;

let transposed = false;

function handleClick() {
    h4 = d3.select('#h4')
    h4.html("Click on a chord to see the decompositons " + (transposed ? "(Focus: Incoming)" : "(Focus: Outgoing)"));
    transposed = !transposed; // Toggle the boolean value
    plotChord() //refresh the page
}

// Get the button element by its ID
const button = document.getElementById('toggleButton');

// Add a click event listener to the button, calling handleClick function
button.addEventListener('click', handleClick);

async function plotChord() {
    try {

        const response = await fetch('./onehot_df.json');

        jsonData = await response.json();

        const sleep = (ms = 2000) => new Promise((r) => setTimeout(r, ms));

        let matrix = [];
        for (let i in jsonData) {
            let row = jsonData[i];
            let t = [];
            for (let j in row) {
                t.push(row[j]);
            }
            matrix.push(t);
        }

        // transpose to switch to more focus on the outgoing connections
        let matrix_T = matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));

        // harcdcoded labels and colors
        let labels = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'];

        let colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
            "#c7c7c7", "#dbdb8d", "#9edae5", "#17becf", "#aec7e8",
            "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
            "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#c7c7c7",
            "#dbdb8d", "#9edae5", "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a",
            "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
            "#dbdb8d", "#9edae5", "#c7c7c7", "#dbdb8d", "#9edae5",
            "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
            "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d",
            "#9edae5", "#c7c7c7", "#dbdb8d", "#9edae5", "#1f77b4",
            "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
            "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
            "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#c7c7c7",
            "#dbdb8d", "#9edae5"
        ];

        //remove previous scg if any
        d3.select("#my_chord").selectAll("svg").remove();

        // create the svg area
        var svg = d3.select("#my_chord")
            .append("svg")
            .attr("width", 1400)
            .attr("height", 1100)
            .append("g")  
            .attr("transform", "translate(700,550)");

        // give this matrix to d3.chord(): it will calculates all the info we need to draw arc and ribbon
        if (transposed) {
            var res = d3.chord()
                .padAngle(0.04)
                (matrix_T)
        } else {
            var res = d3.chord()
                .padAngle(0.04)
                (matrix)
        }

        // To show the tooltip when hovered
        var showTooltip = function (d) {
            var x = d3.event.clientX;
            var y = d3.event.clientY;
            tooltip.style("left", 1100 + "px")
                .style("top", 180 + "px");
            tooltip.show(d);
        }

        // A function that change this tooltip when the leaves a point
        var hideTooltip = function (d) {
            tooltip.hide(d);
        }

        // Create the tooltip
        var tooltip = d3.tip()
            .attr("class", "d3-tip");
        if (transposed) {
            tooltip.html(function (d) {
                return "Source: " + labels[d.source.index] + "<br>Target: " + labels[d.target.index];
            });
        } else {
            tooltip.html(function (d) {
                return "Source: " + labels[d.target.index] + "<br>Target: " + labels[d.source.index];
            });
        }


        // Call the tooltip on the SVG
        svg.call(tooltip);

        // Add the links between groups
        // incoming links are thick i.e. links, of group B, where element A decompses into B
        // outgoing links are thin i.e. links, of group A, where element A decompses into B
        // a bit confusing but can be visualized by the 'U' example
        // to emphasise outgoing links, use matrix_T
        var links = svg
            .datum(res)
            .append("g")
            .selectAll("path")
            .data(function (d) { return d; })
            .enter()
            .append("path")
            .attr("d", d3.ribbon()
                .radius(480)
            )
            .style("fill", function (d) { return colors[d.source.index] })
            .style("stroke", "#00000010")
            .on("click", function (d) {
                links
                    .style("opacity", function (pathData) {
                        return (pathData === d) ? 1.0 : 0.2;
                    })
                    .style("stroke", function (pathData) {
                        return (pathData === d) ? "#00000030" : "#00000000";
                    })
                    .transition()
                    .duration(4000)
                    .on("end", function (d) {
                        links
                            .style("transition", "opacity 1s")
                            .style("opacity", 1.0)
                            .style("stroke", "#00000010")
                    });
            })
            .on("mouseover", showTooltip)
            .on("mouseleave", hideTooltip);

        // this group object uses each group of the data.groups object
        var group = svg
            .datum(res)
            .append("g")
            .selectAll("g")
            .data(function (d) { return d.groups; })
            .enter()

        // add the group arcs on the outer part of the circle
        group.append("g")
            .append("path")
            .style("fill", function (d, i) { return colors[i] })
            .style("stroke", "black")
            .attr("d", d3.arc()
                .innerRadius(500)
                .outerRadius(490)
            )
            .text(function (d) { return labels[d.index]; });

        // Add labels to the group arcs
        group.append("text")
            .each(function (d) { d.angle = (d.startAngle + d.endAngle) / 2; })
            .attr("dy", "0.35em")
            .attr("transform", function (d) {
                return `
      rotate(${(d.angle * 180 / Math.PI - 90)})
      translate(${490 + 15},0)
      ${d.angle > Math.PI ? "rotate(180)" : ""}
    `;
            })
            .style("text-anchor", function (d) { return d.angle > Math.PI ? "end" : null; })
            .text(function (d) { return labels[d.index]; });

    } catch (error) {
        console.error('Error:', error);
    }
}

plotChord();