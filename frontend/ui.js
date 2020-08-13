export function displayPrediction(boredPredictions) {     
    boredPredictions.data().then(data => {
        const prob = data[0];
        console.log(prob);
        console.log(typeof prob);
        
        document.getElementById('boredProb').innerHTML = prob.toFixed(4);
        }
    )
}
