from metaflow import FlowSpec, step, Flow, Parameter, JSONType

class Lab6ClassifierPredictFlow(FlowSpec):
    """ 
    Make predictions on the iris dataset. 
    """
    @step
    def start(self):
        # Access latest run from training flow
        run = Flow('Lab6ClassifierTrainFlow').latest_run

        # Load model and hold-out test data
        self.model = run['end'].task.data.model
        self.test_data = run['end'].task.data.test_data
        self.test_labels = run['end'].task.data.test_labels

        print(f"Loaded model from training flow {run.pathspec}")
        print(f"Test set size: {len(self.test_data)} samples")

        self.next(self.predict)
    
    @step
    def predict(self):
        # Make predictions on the test data
        self.predictions = self.model.predict(self.test_data)
        self.next(self.end)

    @step
    def end(self):
        # Print predictions and ground truth for comparison
        print("Predictions on hold-out set:")
        for i, (pred, label) in enumerate(zip(self.predictions, self.test_labels)):
            print(f"Sample {i}: Predicted = {pred}, Actual = {label}")

if __name__ == '__main__':
    Lab6ClassifierPredictFlow()