package com.logistique.ml;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.LogisticRegressionTrainer;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.MutableDataset;
import org.tribuo.examples.ArrayExample;
import org.tribuo.data.splits.TrainTestSplitter;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;

public class ServicePluie {

    private MutableDataset<Label> train;
    private MutableDataset<Label> test;
    private Model<Label> model;

    public CSVDataSource<Label> chargerJeuDonnees(Path csvPath, RowProcessor<Label> rowProcessor) {
        return new CSVDataSource<>(csvPath, rowProcessor, true);
    }

    public void separerTrainTest(CSVDataSource<Label> dataSource, double proportionTrain, long seed) {
        var splitter = new TrainTestSplitter<>(dataSource, proportionTrain, seed);
        this.train = new MutableDataset<>(splitter.getTrain());
        this.test = new MutableDataset<>(splitter.getTest());
    }

    public Model<Label> entrainer() {
        var trainer = new LogisticRegressionTrainer();
        this.model = trainer.train(train);
        return model;
    }

    public LabelEvaluation evaluer() {
        var evaluator = new LabelEvaluator();
        return evaluator.evaluate(model, test);
    }

    public void sauvegarderModele(Path chemin) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(chemin.toFile()))) {
            oos.writeObject(model);
        }
    }

    @SuppressWarnings("unchecked")
    public Model<Label> chargerModele(Path chemin) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(chemin.toFile()))) {
            return (Model<Label>) ois.readObject();
        }
    }

    public Prediction<Label> predire(Model<Label> modele, String jourSemaine, String retard) {
        Example<Label> exemple = new ArrayExample<>(new Label("non"));
        exemple.add(new Feature("jour_semaine@" + jourSemaine, 1.0));
        exemple.add(new Feature("retard@" + retard, 1.0));
        return modele.predict(exemple);
    }

    public MutableDataset<Label> getTrain() {
        return train;
    }

    public MutableDataset<Label> getTest() {
        return test;
    }

    public Model<Label> getModel() {
        return model;
    }
}
