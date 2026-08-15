External Environment Plugins
============================

Use an external environment plugin when an environment has its own release
cycle or domain dependencies. The environment remains in its owning Python
distribution while RLinf provides the generic loading boundary.

Plugin contract
---------------

Register one entry point in the ``rlinf.envs`` group. The entry-point name is
the value used by ``env.train.env_type`` and ``env.eval.env_type``::

   [project.entry-points."rlinf.envs"]
   my_environment = "my_package.rlinf_plugin:create_plugin"

The loaded object or zero-argument factory must produce an
``rlinf.envs.plugins.EnvPlugin``::

   from rlinf.envs.plugins import EnvPlugin

   from my_package.env import MyEnvironment


   def create_plugin() -> EnvPlugin:
       return EnvPlugin(
           env_cls=MyEnvironment,
           prepare_actions=prepare_actions,
           validate_config=validate_config,
       )

``env_cls`` is required. ``prepare_actions`` and ``validate_config`` are
optional. The action adapter receives the same keyword arguments as
``rlinf.envs.action_utils.prepare_actions``. The validator receives the full
RLinf config, the selected model config, and all train/eval configs that use
the plugin name.

Loading rules
-------------

Built-in environment names keep their current behavior. RLinf consults entry
points only when a name is not built in, requires exactly one provider for
that name, and caches the resulting plugin for the process lifetime. Install
the provider distribution before loading or validating its config; source
checkout path injection is not part of the plugin contract.
