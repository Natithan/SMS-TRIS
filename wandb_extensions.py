import platform
from typing import Union, Optional
import wandb
import os
import shutil
from wandb import util
SYS_PLATFORM = platform.system()
from wandb._globals import _datatypes_callback
from wandb.sdk.lib import filesystem


class NonVersioningWandbImage(wandb.Image):

    def bind_to_run(
        self,
        run: "LocalRun",
        key: Union[int, str],
        step: Union[int, str],
        id_: Optional[Union[int, str]] = None,
        ignore_copy_err: Optional[bool] = None,
    ) -> None:
        # region copied from wandb.Image.bind_to_run
        """Bind this object to a particular Run.

        Calling this function is necessary so that we have somewhere specific to
        put the file associated with this object, from which other Runs can
        refer to it.
        """
        assert self.file_is_set(), "bind_to_run called before _set_file"

        if SYS_PLATFORM == "Windows" and not util.check_windows_valid_filename(key):
            raise ValueError(
                f"Media {key} is invalid. Please remove invalid filename characters"
            )

        # The following two assertions are guaranteed to pass
        # by definition file_is_set, but are needed for
        # mypy to understand that these are strings below.
        assert isinstance(self._path, str)
        assert isinstance(self._sha256, str)

        assert run is not None, 'Argument "run" must not be None.'
        self._run = run

        if self._extension is None:
            _, extension = os.path.splitext(os.path.basename(self._path))
        else:
            extension = self._extension
        # endregion
        #region removed from wandb.Image.bind_to_run
        # if id_ is None:
        #     id_ = self._sha256[:20]
        # file_path = _wb_filename(key, step, id_, extension)
        #endregion
        file_path = f"{str(key)}{extension}"
        # region copied from wandb.Image.bind_to_run
        media_path = os.path.join(self.get_media_subdir(), file_path)
        new_path = os.path.join(self._run.dir, media_path)
        filesystem.mkdir_exists_ok(os.path.dirname(new_path))

        if self._is_tmp:
            shutil.move(self._path, new_path)
            self._path = new_path
            self._is_tmp = False
            _datatypes_callback(media_path)
        else:
            try:
                shutil.copy(self._path, new_path)
            except shutil.SameFileError as e:
                if not ignore_copy_err:
                    raise e
            self._path = new_path
            _datatypes_callback(media_path)
        # endregion