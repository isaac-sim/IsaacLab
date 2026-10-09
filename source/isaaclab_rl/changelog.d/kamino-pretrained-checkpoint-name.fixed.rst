* Fixed ``--checkpoint pretrained`` rejecting tasks that run on a Newton Kamino solver, such as
  ``Isaac-Fourbar-Pole-Swingup``. Newton solvers are named after their config class, for example
  ``newtonkaminopadmm``.
