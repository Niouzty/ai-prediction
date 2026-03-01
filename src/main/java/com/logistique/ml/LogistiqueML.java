package com.logistique.ml;

import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.MutableDataset;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.field.FieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.splits.TrainTestSplitter;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;

public class LogistiqueML {

    private static final LabelFactory LABEL_FACTORY = new LabelFactory();

    public static void preprocessData(Path inputPath, Path outputPath) {
        HeureDepartPreprocessor.convertPreprocessor(inputPath, outputPath);
    }

    public static RowProcessor<Label> buildDefaultRowProcessor() {
        LinkedHashMap<String, FieldProcessor> fieldProcessors = new LinkedHashMap<>();
        fieldProcessors.put("heure_decimal", new DoubleFieldProcessor("heure_decimal"));
        fieldProcessors.put("distance_km", new DoubleFieldProcessor("distance_km"));
        fieldProcessors.put("pluie", new IdentityProcessor("pluie"));
        fieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        fieldProcessors.put("vehicule_type", new IdentityProcessor("vehicule_type"));

        FieldResponseProcessor<Label> responseProcessor =
                new FieldResponseProcessor<>("retard", "non", LABEL_FACTORY);
        return new RowProcessor<>(responseProcessor, fieldProcessors);
    }

    public static CSVDataSource<Label> loadDataSource(Path filePath, RowProcessor<Label> processor) {
        return new CSVDataSource<>(filePath, processor, true);
    }

    public static TrainTestSplitter<Label> createTrainTestSplitter(CSVDataSource<Label> source,
                                                                   double trainRatio,
                                                                   long seed) {
        return new TrainTestSplitter<>(source, trainRatio, seed);
    }

    public static Model<Label> trainModel(MutableDataset<Label> trainData) {
        return new LogisticRegressionTrainer().train(trainData);
    }

    public static LabelEvaluation evaluateModel(Model<Label> model, MutableDataset<Label> testData) {
        return new LabelEvaluator().evaluate(model, testData);
    }

    public static void saveModel(Model<Label> model, Path targetPath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(targetPath.toFile()))) {
            out.writeObject(model);
        }
    }

    @SuppressWarnings("unchecked")
    public static Model<Label> loadModel(Path sourcePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(sourcePath.toFile()))) {
            return (Model<Label>) in.readObject();
        }
    }

    public static Prediction<Label> predict(Model<Label> model, Example<Label> example) {
        return model.predict(example);
    }
}
