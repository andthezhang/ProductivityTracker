export function displayPrediction(boredPredictions) {     
    boredPredictions.data().then(data => {
        const prob = data[0];
        console.log(prob);
        console.log(typeof prob);
        
        document.getElementById('boredProb').innerHTML = prob.toFixed(4);
        }
    )
}
var chart = new CanvasJS.Chart("chartContainer", {
	title: {
		text: "Your Productivity Level"
	},
	axisY: {
        title: "Probability of being productive",
        maximum: 100,
		includeZero: true,
		suffix: " %"
	},
	data: [{
		type: "column",	
		indexLabel: "{y}",
		dataPoints: [
            { label: "Productive Level", y: 50 },
		]
	}]
});
export async function updateChart(boredPredictions) {
    boredPredictions.data().then(data => {
        const prob = data[1].toFixed(4)*100;
        const newColor = prob <= 30 ? "#FF2500" : prob <= 50 ? "#FF6000" : prob > 50 ? "#6B8E23 " : null;
        chart.options.data[0].dataPoints = [{label: "Productive Level", y: prob, color: newColor}]
        chart.render();
        }
    )
};
