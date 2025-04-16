from metaflow import FlowSpec, step, Flow, Parameter, JSONType

class Lab6ClassifierPredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True)

    @step
    def start(self):
        run = Flow('Lab6ClassifierTrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        print("Input vector", self.vector)
        self.next(self.end)

    @step
    def end(self):
        print('Model', self.model)
        print('Predicted class', self.model.predict([self.vector])[0])

if __name__=='__main__':
    Lab6ClassifierPredictFlow()