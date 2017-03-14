package Neuralnet;

public class Network {
	Layer[] layer;
	Layer inputLayer;
	Layer outputLayer;
	/**
	 * 
	 * @param inputs -> input layers
	 * @param outputs -> output layers
	 * @param hiddens -> hidden layers
	 */
	public Network(int inputs, int[] hiddens, int outputs){
		layer = new Layer[hiddens.length +2];
		inputLayer = new Layer(inputs, 0);
		layer[0] = inputLayer;
		//Size of previous layer
		int previous = inputs;
		//iterate over all the hidden layers and add them;
		for(int h = 0; h< hiddens.length; h++){
			layer[h+1] = new Layer(hiddens[h], previous);
			previous = hiddens[h];
		}
		//add output
		outputLayer = new Layer(outputs, previous);
		layer[layer.length-1] = outputLayer;
	}
	//take in inputs, compute output
	public double[] compute(double[]inputs){
		//Put the input values into the input neurons
		for(int i = 0; i< inputLayer.size();i++){
			inputLayer.getNeuron(i).setValue(inputs[i]);
		}
		//Reference to previous layer
		Layer previousLayer = inputLayer;
		//Go to each layer
		for(Layer currentLayer:layer){
			//Go to each neuron in this layer
			for(int n = 0; n< currentLayer.size();n++){
				//The current Neuron
				Neuron currentNeuron = currentLayer.getNeuron(n);
				//calculate sigmod
				double sum = 0;
				//go to each previous neuron
				for(int w = 0; w<currentNeuron.size();w++){
					
					Neuron previousNeuron = previousLayer.getNeuron(w);
					double neuronWeight = currentNeuron.getWeight(w);
					//add the weighted value to the sum
					sum += neuronWeight*previousNeuron.getValue();
				}
				//Set the current neuron's value to the sigmoid sum
				currentNeuron.setValue(sigmoid(sum));
				
			}
		}
		// copy all the output values to an array
		double[] output = new double[outputLayer.size()];
		for(int i = 0; i< outputLayer.size(); i++){
			output[i]= outputLayer.getNeuron(i).getValue();
		}
		return output;
	}
	/**
	 * Adjust the weights of each connection to get closer to correct output
	 * @param expected
	 * @param learningRate-how fast to learn
	 */
	public void propagate(double[]expected, double learningRate){
		Layer previousLayer= layer[layer.length-2];
		//Calculate the weights
		for(int n = 0; n<outputLayer.size();n++){
			Neuron outputNeuron = outputLayer.getNeuron(n);
			//the value and error for the output neuron
			double value = outputNeuron.getValue();
			double error = expected[n] - outputNeuron.getValue();
			for(int w = 0; w<outputNeuron.size();w++){
				double previousValue = previousLayer.getNeuron(w).getValue();
				double derivative = -value * (1-value)*previousValue*error;
				double newWeight = outputNeuron.getWeight(w)-derivative*learningRate;
				outputNeuron.setWeight(w,newWeight);
			}
		}
		//for every hidden layer
		for(int l = layer.length-2;l>=1;l--){
			Layer currentLayer = layer[l];
			Layer nextLayer = layer[l+1];
			previousLayer = layer[l-1];
			for(int n=0;n<currentLayer.size();n++){
				//get the neuron and the value
				Neuron currentNeuron = currentLayer.getNeuron(n);
				double value = currentNeuron.getValue();
				for(int w = 0; w<currentNeuron.size();w++){
					double previousValue = previousLayer.getNeuron(w).getValue();
					double sum = 0;
					
					//add to the sum
					for(int i = 0; i<nextLayer.size();i++){
						Neuron nextNeuron = nextLayer.getNeuron(i);
						double nextWeight = nextNeuron.getWeight(n);
						double nextValue = nextNeuron.getValue();
					}
					double derivative = value*(1-value)*previousValue*sum;
					double newWeight = currentNeuron.getWeight(w)-learningRate*derivative;
					currentNeuron.setWeight(w, newWeight);
				}
			}
		}
	}
	public double sigmoid(double value){
		return 1.0/(1.0-Math.exp(-value));
	}
	
}

