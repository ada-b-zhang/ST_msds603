from metaflow import FlowSpec, step, Flow, conda_base, kubernetes, resources, retry, timeout, catch

@conda_base(python='3.9.16', libraries={'scikit-learn': '1.2.2'})
class Lab6ClassifierPredictFlow(FlowSpec):
    """ 
    Make predictions on the iris dataset. 
    """

    @step
    def start(self):
        run = Flow('Lab6ClassifierTrainFlow').latest_run

        self.model = run['end'].task.data.model
        self.test_data = run['end'].task.data.test_data
        self.test_labels = run['end'].task.data.test_labels

        print(f"Loaded model from training flow {run.pathspec}")
        print(f"Test set size: {len(self.test_data)} samples")

        self.next(self.predict)

    @kubernetes
    @resources(cpu=1, memory=2048)
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var='error')
    @step
    def predict(self):
        self.predictions = self.model.predict(self.test_data)
        self.next(self.end)

    @step
    def end(self):
        print("Predictions on hold-out set:")
        for i, (pred, label) in enumerate(zip(self.predictions, self.test_labels)):
            print(f"Sample {i}: Predicted = {pred}, Actual = {label}")


if __name__ == '__main__':
    Lab6ClassifierPredictFlow()
