import pickle
import numpy as np
from os.path import join
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precicion

from utils.modules import get_model


class Solver(object):
    def __init__(self, args, data_loader, valid_loader=None):
        self.data_loader = data_loader
        self.valid_loader = valid_loader

        self.mode = args.mode
        self.batch_size = args.batch_size
        self.mixed_training = args.mixed_training
        self.n_epochs = args.n_epochs
        self.save_dir = args.save_dir

        if args.mixed_training:
            policy = mixed_precicion.Policy('mixed_float16')
            mixed_precicion.set_policy(policy)

        self.n_classes = len(np.unique(data_loader.y_train))
        self.model = get_model((None, None, 3), self.n_classes)
        self.model.compile(
            loss=[losses.SparseCategoricalCrossentropy(),
                  losses.SparseCategoricalCrossentropy()],
            optimizer=optimizers.Adam(lr=args.lr),
            metrics = ['acc']
        )

    def train(self):
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath=join(self.save_dir, "w.hdf5"),
                monitor="val_perception_branch_output_acc",
                save_best_only=True,
                mode="max",
                verbose=1
            )
        )
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_perception_branch_output_acc",
                factor=.5,
                patience=2,
                min_lr=1e-6,
                mode='max',
                verbose=1
            )
        )

        history = self.model.fit_generator(
            generator=self.data_loader,
            validation_data=self.valid_loader,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.data_loader),
            validation_steps=len(self.valid_loader),
            verbose=1,
            workers=10,
            max_queue_size=30,
            callbacks=callbacks
        )

        with open(join(self.save_dir, "train_log.pkl"), "wb") as f:
            pickle.dump(history.hitory, f)

