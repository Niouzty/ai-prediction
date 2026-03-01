package com.logistique.ml;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.examples.ArrayExample;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelEvaluation;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.MutableDataset;
import org.tribuo.data.columnar.RowProcessor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests d'implémentation du Machine Learning pour prédire les retards de livraison
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LogistiqueLMIATests {

    private static final String fileName = "livraison_retards_dataset.csv";
    private static final String newFileName = "livraison_retards_dataset_converted.csv";
    private static final String modelFile = "livraison_regressor.ser";
    private static final Path input = Paths.get("src", "main", "resources", fileName);
    private static final Path output = Paths.get("src", "main", "resources", newFileName);
    private static final Path MODEL_PATH = Paths.get("src", "main", "resources", modelFile);

    private static RowProcessor<Label> rowProcessor;
    private static CSVDataSource<Label> dataSource;
    private static MutableDataset<Label> train;
    private static MutableDataset<Label> test;
    private static Model<Label> model;
    private static Prediction<Label> prediction;

    @BeforeAll
    public static void setUp() {
        rowProcessor = LogistiqueML.buildDefaultRowProcessor();
        System.out.println("✅ Initialisation effectuée");
    }

    @AfterAll
    public static void tearDown() {
        if (output.toFile().exists()) {
            boolean deleted = output.toFile().delete();
            assertTrue(deleted, "Le fichier converti doit être supprimé après les tests");
        }
        if (MODEL_PATH.toFile().exists()) {
            boolean deleted = MODEL_PATH.toFile().delete();
            assertTrue(deleted, "Le fichier du modèle doit être supprimé après les tests");
        }
        System.out.println("✅ Nettoyage des ressources effectué");
    }

    /**
     * Test 1: Prétraitement - Convertir heure_depart en format numérique
     */
    @Test
    @Order(1)
    void prepareDatasets() throws IOException {
        System.out.println("\n📋 Étape 1: Prétraitement des données");
        LogistiqueML.preprocessData(input, output);
        assertTrue(output.toFile().exists(), "Le fichier converti doit exister");
        assertTrue(output.toFile().length() > 0, "Le fichier converti ne doit pas être vide");
        System.out.println("✅ Fichier prétraité avec succès");
    }

    /**
     * Test 2: Chargement des données
     */
    @Test
    @Order(2)
    void loadDatasets() throws IOException {
        System.out.println("\n📂 Étape 2: Chargement des données");
        dataSource = LogistiqueML.loadDataSource(Paths.get("src", "main", "resources", newFileName), rowProcessor);
        assertNotNull(dataSource, "La source de données ne doit pas être null");
        System.out.println("✅ Données chargées avec succès");
        System.out.println("   Nombre d'exemples: " + dataSource.size());
    }

    /**
     * Test 3: Division Train/Test (80% / 20%)
     */
    @Test
    @Order(3)
    void splitTrainTest() {
        System.out.println("🔀 Étape 3: Division Train/Test");
        var splitter = LogistiqueML.createTrainTestSplitter(dataSource, 0.8, 42L);
        train = new MutableDataset<>(splitter.getTrain());
        test = new MutableDataset<>(splitter.getTest());

        assertNotNull(train, "L'ensemble d'entraînement ne doit pas être null");
        assertNotNull(test, "L'ensemble de test ne doit pas être null");
        assertTrue(train.size() > 0, "L'ensemble d'entraînement doit contenir des données");
        assertTrue(test.size() > 0, "L'ensemble de test doit contenir des données");

        System.out.println("✅ Division effectuée");
        System.out.println("   Train: " + train.size() + " exemples");
        System.out.println("   Test: " + test.size() + " exemples");
    }

    /**
     * Test 4: Entraînement du modèle
     */
    @Test
    @Order(4)
    void training() {
        System.out.println("🏋️ Étape 4: Entraînement du modèle");
        model = LogistiqueML.trainModel(train);

        assertNotNull(model, "Le modèle ne doit pas être null");
        System.out.println("✅ Modèle entraîné avec succès");
        System.out.println("   Modèle: LogisticRegression");
        System.out.println("   Nombre de features: " + model.getFeatureIDMap().size());
    }

    /**
     * Test 5: Évaluation du modèle
     */
    @Test
    @Order(5)
    void evaluator() {
        System.out.println("📊 Étape 5: Évaluation du modèle");
        LabelEvaluation evaluation = LogistiqueML.evaluateModel(model, test);

        System.out.println("✅ Résultats d'évaluation:");
        System.out.println(evaluation.toString());
    }

    /**
     * Test 6: Sauvegarde du modèle
     */
    @Test
    @Order(6)
    void saveModel() throws Exception {
        System.out.println("💾 Étape 6: Sauvegarde du modèle");
        LogistiqueML.saveModel(model, MODEL_PATH);

        assertTrue(MODEL_PATH.toFile().exists(), "Le fichier du modèle doit exister");
        System.out.println("✅ Modèle sauvegardé avec succès");
        System.out.println("   Fichier: " + MODEL_PATH);
    }

    /**
     * Test 7: Prédiction avec le modèle chargé
     */
    @Test
    @Order(7)
    void predictor() throws Exception {
        System.out.println("🔮 Étape 7: Prédiction");
        Model<Label> loadedModel = LogistiqueML.loadModel(MODEL_PATH);

        // Créer un exemple pour tester
        var example = new ArrayExample<>(new Label("non"));
        example.add(new Feature("distance_km@value", 120.0));
        example.add(new Feature("heure_decimal@value", 8.0));
        example.add(new Feature("pluie@non", 1.0)); // pluie: non
        example.add(new Feature("pluie@oui", 0.0));
        example.add(new Feature("jour_semaine@lundi", 0.0));
        example.add(new Feature("jour_semaine@mardi", 0.0));
        example.add(new Feature("jour_semaine@mercredi", 1.0)); // jour: mercredi
        example.add(new Feature("jour_semaine@jeudi", 0.0));
        example.add(new Feature("jour_semaine@vendredi", 0.0));
        example.add(new Feature("vehicule_type@camion", 0.0));
        example.add(new Feature("vehicule_type@camionnette", 1.0)); // type: camionnette
        example.add(new Feature("vehicule_type@fourgon", 0.0));

        // Prédiction
        prediction = LogistiqueML.predict(loadedModel, example);

        assertNotNull(prediction, "La prédiction ne doit pas être null");
        System.out.println("✅ Prédiction effectuée");
        System.out.println("   Paramètres: distance=120km, heure=8:00, pluie=non, jour=mercredi, type=camionnette");
        System.out.println("   🎯 Résultat: " + prediction.getOutput());
        System.out.println("   Probabilités: " + prediction.getOutputScores());
    }

    /**
     * Test BONUS: Créer un modèle pour prédire la pluie
     * À partir des données: jour_semaine et retard => prédire pluie
     */
    @Test
    @Order(8)
    void bonusPredictor() throws Exception {
        System.out.println("⭐ BONUS: Prédiction de la PLUIE");
        System.out.println("Modèle: jour_semaine + retard => pluie");

        // Créer les fieldProcessors pour le bonus
        LinkedHashMap<String, FieldProcessor> bonusFieldProcessors = new LinkedHashMap<>();
        bonusFieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        bonusFieldProcessors.put("retard", new IdentityProcessor("retard"));

        // Le label à prédire est maintenant "pluie"
        LabelFactory bonusLabelFactory = new LabelFactory();
        FieldResponseProcessor<Label> bonusResponseProcessor =
                new FieldResponseProcessor<>("pluie", "non", bonusLabelFactory);
        RowProcessor<Label> bonusRowProcessor = new RowProcessor<>(bonusResponseProcessor, bonusFieldProcessors);

        // Charger les données avec les nouveaux processors
        CSVDataSource<Label> bonusDataSource = new CSVDataSource<>(
                Paths.get("src", "main", "resources", newFileName),
                bonusRowProcessor,
                true
        );

        // Split train/test
        var bonusSplitter = new TrainTestSplitter<>(bonusDataSource, 0.8, 42L);
        MutableDataset<Label> bonusTrain = new MutableDataset<>(bonusSplitter.getTrain());
        MutableDataset<Label> bonusTest = new MutableDataset<>(bonusSplitter.getTest());

        System.out.println("✅ Données chargées pour la prédiction de pluie");
        System.out.println("   Train: " + bonusTrain.size() + " exemples");
        System.out.println("   Test: " + bonusTest.size() + " exemples");

        // Entraînement du modèle pour prédire la pluie
        var bonusTrainer = new LogisticRegressionTrainer();
        Model<Label> bonusModel = bonusTrainer.train(bonusTrain);

        System.out.println("✅ Modèle de prédiction de pluie entraîné");

        // Évaluation
        var bonusEvaluator = new LabelEvaluator();
        LabelEvaluation bonusEvaluation = bonusEvaluator.evaluate(bonusModel, bonusTest);

        System.out.println("\n📊 Résultats d'évaluation (Pluie):");
        System.out.println(bonusEvaluation.toString());

        // Prédiction sur un exemple : vendredi avec retard
        System.out.println("\n🔮 Prédiction de pluie");
        var bonusExample = new ArrayExample<>(new Label("non"));
        bonusExample.add(new Feature("jour_semaine@lundi", 0.0));
        bonusExample.add(new Feature("jour_semaine@mardi", 0.0));
        bonusExample.add(new Feature("jour_semaine@mercredi", 0.0));
        bonusExample.add(new Feature("jour_semaine@jeudi", 0.0));
        bonusExample.add(new Feature("jour_semaine@vendredi", 1.0)); // vendredi
        bonusExample.add(new Feature("retard@non", 0.0));
        bonusExample.add(new Feature("retard@oui", 1.0)); // retard: oui

        Prediction<Label> bonusPrediction = bonusModel.predict(bonusExample);

        System.out.println("   Paramètres: jour=vendredi, retard=oui");
        System.out.println("   🎯 Prédiction pluie: " + bonusPrediction.getOutput());
        System.out.println("   Probabilités: " + bonusPrediction.getOutputScores());
    }

    /**
     * Test 9: Service structuré pour la prédiction de pluie
     */
    @Test
    @Order(9)
    void servicePluie() throws Exception {
        System.out.println("\n☔️ Étape 9: Service structuré pour prédiction de pluie");

        ModelePluie modelePluie = new ModelePluie();
        ServicePluie service = new ServicePluie();
        AnalysePredictions analyse = new AnalysePredictions();

        var bonusDataSource = service.chargerJeuDonnees(Paths.get("src", "main", "resources", newFileName), modelePluie.obtenirRowProcessor());
        service.separerTrainTest(bonusDataSource, 0.8, 42L);
        Model<Label> pluieModel = service.entrainer();

        assertNotNull(pluieModel, "Le modèle pluie ne doit pas être null");

        LabelEvaluation pluieEvaluation = service.evaluer();
        analyse.afficherRapportEvaluation(pluieEvaluation);

        var probabilityEntries = analyse.construireArbreProbabilite(pluieModel, service.getTest());
        assertFalse(probabilityEntries.isEmpty(), "La liste des probabilités doit contenir des entrées");
        analyse.afficherArbreProbabilite(probabilityEntries);

        Path rainModelPath = Paths.get("src", "main", "resources", "pluie_model.ser");
        service.sauvegarderModele(rainModelPath);
        assertTrue(Files.exists(rainModelPath), "Le modèle pluie sauvegardé doit exister");

        Model<Label> loadedRainModel = service.chargerModele(rainModelPath);
        Prediction<Label> rainPrediction = service.predire(loadedRainModel, "vendredi", "oui");

        analyse.afficherPrediction(rainPrediction, "jour_semaine=vendredi, retard=oui");
        assertNotNull(rainPrediction, "La prédiction pluie ne doit pas être null");

        Files.deleteIfExists(rainModelPath);
    }
}
