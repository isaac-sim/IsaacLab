@ECHO OFF

pushd %~dp0

REM Command file to build Sphinx documentation

set SOURCEDIR=.
set BUILDDIR=_build

REM Check if a specific target was passed
if "%1" == "multi-docs" (
	REM Check if SPHINXBUILD is set, if not default to sphinx-multiversion
	if "%SPHINXBUILD%" == "" (
		set SPHINXBUILD=sphinx-multiversion
	)
	%SPHINXBUILD% >NUL 2>NUL
	if errorlevel 9009 (
		echo.
		echo.The 'sphinx-multiversion' command was not found. Make sure you have Sphinx
		echo.installed, then set the SPHINXBUILD environment variable to point
		echo.to the full path of the 'sphinx-multiversion' executable. Alternatively you
		echo.may add the Sphinx directory to PATH.
		echo.
		echo.If you don't have Sphinx installed, grab it from
		echo.http://sphinx-doc.org/
		exit /b 1
	)
	%SPHINXBUILD% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

	REM Copy the redirect index.html to the build directory
	copy _redirect\index.html %BUILDDIR%\index.html
	goto end
)

if "%1" == "current-docs" (
	REM Check if SPHINXBUILD is set, if not default to sphinx-build
	if "%SPHINXBUILD%" == "" (
		set SPHINXBUILD=sphinx-build
	)
	%SPHINXBUILD% >NUL 2>NUL
	if errorlevel 9009 (
		echo.
		echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
		echo.installed, then set the SPHINXBUILD environment variable to point
		echo.to the full path of the 'sphinx-build' executable. Alternatively you
		echo.may add the Sphinx directory to PATH.
		echo.
		echo.If you don't have Sphinx installed, grab it from
		echo.http://sphinx-doc.org/
		exit /b 1
	)
	%SPHINXBUILD% %SOURCEDIR% %BUILDDIR%\current %SPHINXOPTS% %O%
	goto end
)

REM If no valid target is passed, show usage instructions
echo.
echo.Usage:
echo.  make.bat multi-docs    - To build the multi-version documentation.
echo.  make.bat current-docs  - To build the current documentation.
echo.

:end
popd
