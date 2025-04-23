from metaflow import FlowSpec, step, conda_base, kubernetes, resources, retry, timeout, catch

# @conda_base(python='3.9.16', libraries={'scikit-learn': '1.2.2', 'mlflow': '2.3.1'})
@conda_base(python='3.9.16', libraries={
    'scikit-learn': '1.2.2',
    'mlflow': '2.3.1',
    'databricks-cli': '0.17.6'  
})
class Lab6ClassifierTrainFlow(FlowSpec):
    """ 
    Train the classifier for iris dataset. 
    """

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        X, y = datasets.load_iris(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, random_state=0)
        print("Data loaded successfully, yay!")
        self.next(self.train_rf_classifier, self.train_adaboost_classifier)

    @kubernetes
    @resources(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=600)
    @catch(var='error')
    @step
    def train_rf_classifier(self):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @kubernetes
    @resources(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=600)
    @catch(var='error')
    @step
    def train_adaboost_classifier(self):
        from sklearn.ensemble import AdaBoostClassifier

        self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri('https://msds603servicename-1092058745718.us-west2.run.app')
        mlflow.set_experiment('metaflow-lab6')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]

        self.test_data = inputs[0].test_data
        self.test_labels = inputs[0].test_labels

        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path='metaflow_train', registered_model_name="metaflow-iris-model")

        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % (type(res[0]).__name__, res[1]) for res in self.results))
        print('Best model:', type(self.model).__name__)
        print('Test data shape:', len(self.test_data), 'labels:', len(self.test_labels))


if __name__ == '__main__':
    Lab6ClassifierTrainFlow()
