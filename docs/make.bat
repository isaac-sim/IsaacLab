@ECHO OFF

pushd %~dp0

REM Command file to build Sphinx documentation

set SOURCEDIR=.
set BUILDDIR=_build
if "%DOCS_DEFAULT_REF%" == "" set DOCS_DEFAULT_REF=main

REM Check if a specific target was passed
if "%1" == "multi-docs" (
	REM Check if SPHINXBUILD is set, if not default to sphinx-multiversion
	if "%SPHINXBUILD%" == "" (
		set SPHINXBUILD=sphinx-multiversion
	)
	where %SPHINXBUILD% >NUL 2>NUL
	if errorlevel 1 (
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
	if errorlevel 1 exit /b 1

	if not exist "%BUILDDIR%\%DOCS_DEFAULT_REF%\index.html" (
		echo Default docs ref '%DOCS_DEFAULT_REF%' was not built
		exit /b 1
	)

	REM Render the redirect index.html using the selected default docs ref
	powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $template = [System.IO.File]::ReadAllText('_redirect\index.html'); $output = $template.Replace('__DOCS_DEFAULT_REF__', $env:DOCS_DEFAULT_REF); [System.IO.File]::WriteAllText('%BUILDDIR%\index.html', $output); exit 0"
	if errorlevel 1 exit /b 1
	popd
	exit /b 0
)

if "%1" == "current-docs" (
	REM Check if SPHINXBUILD is set, if not default to sphinx-build
	if "%SPHINXBUILD%" == "" (
		set SPHINXBUILD=sphinx-build
	)
	where %SPHINXBUILD% >NUL 2>NUL
	if errorlevel 1 (
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
	if exist "%BUILDDIR%\current" rmdir /s /q "%BUILDDIR%\current"
	%SPHINXBUILD% -W "%SOURCEDIR%" "%BUILDDIR%\current" %SPHINXOPTS%
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
exit /b 0
