import matplotlib.pyplot as plt
from linesight.exceptions import LineSightNotFittedError


class LineSightBase:

    _is_fitted: bool = False

    def _check_fitted(self, method_name: str) -> None:
        if not self._is_fitted:
            raise LineSightNotFittedError(
                f"Call fit(X, y) before calling {method_name}()."
            )

    def _validate_hyperparams(self) -> None:
        """Hook for subclasses to validate hyperparams before fitting."""
        if hasattr(self, 'learning_rate') and getattr(self, 'learning_rate') <= 0:
            raise ValueError("learning_rate must be > 0")
        if hasattr(self, 'epochs') and getattr(self, 'epochs') <= 0:
            raise ValueError("epochs must be > 0")
        if hasattr(self, 'alpha') and getattr(self, 'alpha') < 0:
            raise ValueError("alpha must be >= 0")
        if hasattr(self, 'l1_ratio') and not (0 <= getattr(self, 'l1_ratio') <= 1):
            raise ValueError("l1_ratio must be between 0 and 1")
        if hasattr(self, 'degree') and getattr(self, 'degree') < 1:
            raise ValueError("degree must be >= 1")
        if hasattr(self, 'batch_size'):
            bs = getattr(self, 'batch_size')
            if bs is not None and bs <= 0:
                raise ValueError("batch_size must be >= 1 or None")

    def show(self, animation_obj=None, fig=None):
        """
        Display a figure or animation correctly for the current environment.

        THE CORRECT BEHAVIOR — read carefully:

        For STATIC figures (fig parameter):
          - In Jupyter/Colab: plt.show() closes the figure and triggers
            the inline backend to render it. Do NOT return fig after plt.show()
            because the figure object is now closed. Return None.
          - In script: plt.show() opens the window. Return None.
          - NEVER call IPython.display.display(fig) — causes triple rendering.

        For ANIMATIONS (animation_obj parameter):
          - In Jupyter/Colab: return HTML(anim.to_jshtml()).
            Do NOT call plt.show() first — it closes the figure.
          - In script: plt.show() runs the animation in a window.

        This corrects the bug in the original specification where static
        figures returned fig after plt.show() had already closed them.
        """
        from linesight.utils.environment import _detect_environment
        env = _detect_environment()

        if animation_obj is not None:
            if env in ('jupyter', 'colab'):
                try:
                    from IPython.display import HTML
                    plt.close()   # close the underlying figure to free memory
                    return HTML(animation_obj.to_jshtml())
                except ImportError:
                    plt.show()
                    return None
            else:
                plt.show()
                return None

        elif fig is not None:
            # For static figures: plt.show() handles both environments.
            # In Jupyter with %matplotlib inline, plt.show() triggers the
            # inline renderer. In scripts it opens the window.
            # After plt.show(), the figure is closed. Return None always.
            plt.tight_layout()
            plt.show()
            return None

        return None

    def save(self, filepath: str, fig=None, animation_obj=None,
             dpi: int = 150, fps: int = 20) -> str:
        """
        Save a figure or animation to disk.

        Parameters
        ----------
        filepath : str
            Full path including extension.
            For figures: use .png, .pdf, .svg
            For animations: use .gif or .mp4
            Example: "my_plot.png" or "training.gif"
        fig : matplotlib.figure.Figure, optional
        animation_obj : FuncAnimation, optional
        dpi : int, default 150
            Resolution for raster formats (png, gif).
        fps : int, default 20
            Frames per second for animation exports.

        Returns
        -------
        str — the filepath that was saved to, for confirmation.

        Usage in visualization methods
        --------------------------------
        To let users save without displaying:
            fig = model.plot_fit(X, y, display=False)
            model.save("fit.png", fig=fig)

        Or as a one-liner (display=False returns the object):
            model.save("training.gif",
                       animation_obj=model.animate_training(X, y, display=False))

        Raises
        ------
        ValueError if neither fig nor animation_obj is provided.
        ImportError with helpful message if saving .mp4 requires ffmpeg.
        """
        if fig is None and animation_obj is None:
            raise ValueError(
                "Pass either fig= or animation_obj= to save().\n"
                "Get the object by calling the visualization method "
                "with display=False."
            )

        if fig is not None:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved to: {filepath}")
            return filepath

        if animation_obj is not None:
            if filepath.endswith('.gif'):
                animation_obj.save(filepath, writer='pillow', fps=fps)
            elif filepath.endswith('.mp4'):
                try:
                    animation_obj.save(filepath, writer='ffmpeg', fps=fps)
                except Exception:
                    raise ImportError(
                        "Saving .mp4 requires ffmpeg installed on your system.\n"
                        "Install it with: conda install ffmpeg\n"
                        "Or save as .gif instead: model.save('training.gif', ...)"
                    )
            else:
                animation_obj.save(filepath, fps=fps)
            print(f"Saved to: {filepath}")
            return filepath

    def summary(self) -> str:
        """
        Print a complete model summary: parameters, fit quality, coefficients.

        Every regression class overrides this to include model-specific info.
        The base implementation raises NotFittedError if called before fit().

        Returns
        -------
        str — the summary text. Also printed (in script mode) or returned
        (in Jupyter, displayed automatically as the cell's last value).
        """
        self._check_fitted("summary")
        raise NotImplementedError(
            "summary() must be implemented by each regression subclass."
        )

    def refit(self, X, y):
        """
        Re-train the model from scratch on new data.

        Resets ALL state: coefficients, intercept, history, fitted flag.
        Equivalent to creating a new instance with the same hyperparameters
        and calling fit(X, y).

        Why this exists
        ---------------
        If a student calls fit() twice on the same model object, the second
        call continues from where the first left off (coef_ is NOT reset to 0).
        This produces confusing results. refit() makes the intention explicit
        and resets properly.

        Usage
        -----
        model.fit(X_train, y_train)        # initial training
        model.refit(X_new, y_new)          # clean reset + retrain on new data
        """
        # Reset state — subclasses that use theta_ instead of coef_/intercept_
        # must override this to reset theta_ as well
        if hasattr(self, 'coef_'):
            self.coef_ = 0.0
        if hasattr(self, 'intercept_'):
            self.intercept_ = 0.0
        if hasattr(self, 'theta_'):
            self.theta_ = None
        self._is_fitted = False
        from linesight.utils.history import TrainingHistory
        self._history = TrainingHistory(learning_rate=getattr(self, 'learning_rate', 0))
        return self.fit(X, y)
