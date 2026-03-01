package com.logistique.ml;

import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class AnalysePredictions {

    private static final Logger logger = Logger.getLogger(AnalysePredictions.class.getName());

    public record EntreeProbabilite(int index, Label prediction, Map<String, Double> scores) {}

    public List<EntreeProbabilite> construireArbreProbabilite(Model<Label> model, MutableDataset<Label> testData) {
        List<EntreeProbabilite> entrees = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            var exemple = testData.getExample(i);
            var prediction = model.predict(exemple);
            entrees.add(new EntreeProbabilite(i, prediction.getOutput(), prediction.getOutputScores()));
        }
        return entrees;
    }

    public void afficherRapportEvaluation(LabelEvaluation evaluation) {
        logger.info("=== Résultats de l'évaluation ===");
        logger.info(evaluation.toString());
        logger.info("Précision : " + evaluation.accuracy());
        logger.info("Matrice de confusion :\n" + evaluation.getConfusionMatrix().toString());
    }

    public void afficherArbreProbabilite(List<EntreeProbabilite> entrees) {
        logger.info("=== Arbre des probabilités ===");
        for (var entree : entrees) {
            logger.info(String.format("Exemple %d -> Prédiction: %s | Scores: %s",
                    entree.index(), entree.prediction(), entree.scores()));
        }
    }

    public void afficherPrediction(Prediction<Label> prediction, String description) {
        logger.info("=== Résultat de la prédiction ===");
        logger.info("Entrée : " + description);
        logger.info("Prédiction : " + prediction.getOutput());
        logger.info("Scores : " + prediction.getOutputScores());
    }
}
